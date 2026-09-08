import isaacgym
import torch
import numpy as np
import os
import json
from isaacgym import gymapi, gymtorch

class TrueEmbodiedNavEnv:
    def __init__(self, num_envs=20, device="cuda:0"):
        self.num_envs = num_envs
        self.device = device
        self.use_roboshape = False
        self.privacy_task = False
        self.env_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.start_x = torch.zeros(self.num_envs, device=self.device)
        
    def set_task(self, use_roboshape: bool, privacy_task: bool):
        self.use_roboshape = use_roboshape
        self.privacy_task = privacy_task
        
        # Recompute obs_dim
        embed_dim = 512 if not self.use_roboshape else 64
        self.num_obs = embed_dim + 3 # + robot_pos only
        
        self.reset()
        
    def initialize(self):
        # Force PyTorch CUDA context initialization BEFORE Isaac Gym starts its own context
        _dummy = torch.zeros(1, device=self.device)
        
        self.gym = gymapi.acquire_gym()
        
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.dt = 1.0 / 60.0
        sim_params.substeps = 2
        sim_params.use_gpu_pipeline = True
        sim_params.physx.use_gpu = True
        print("Creating sim...", flush=True)
      
        self.sim = self.gym.create_sim(0, -1, gymapi.SIM_PHYSX, sim_params)
        print("Sim created successfully!", flush=True)
        
        self.scans_dir = "/pathto/scans"
        all_scenes = [d for d in os.listdir(self.scans_dir) if os.path.isdir(os.path.join(self.scans_dir, d)) and d.startswith("scene")]
        valid_scenes = []
        for d in all_scenes:
            if os.path.exists(os.path.join(self.scans_dir, d, f"{d}.obj")):
                valid_scenes.append(d)
        
        self.scenes = valid_scenes[:self.num_envs]
        if len(self.scenes) < self.num_envs:
            print(f"Warning: Only found {len(self.scenes)} valid meshes, but requested {self.num_envs} envs. Repeating meshes.")
            while len(self.scenes) < self.num_envs:
                self.scenes.append(self.scenes[-1])
        
        # Load the scene type registries (both subset1 and subset2)
        self.scene_to_is_bedroom = {}
        try:
            with open("/pathto/scenelog1.json", "r") as f:
                scene_log1 = json.load(f)
                for item in scene_log1:
                    self.scene_to_is_bedroom[item["scene_name"]] = item["is_bedroom"]
        except Exception as e:
            print(f"Warning: Could not load subset1 registry. {e}")
            
        try:
            with open("pathto/scene_log.json", "r") as f:
                scene_log2 = json.load(f)
                for item in scene_log2:
                    self.scene_to_is_bedroom[item["scene_name"]] = item["is_bedroom"]
        except Exception as e:
            print(f"Warning: Could not load subset2 registry. {e}")
        
        self.spatial_embeddings = {}
        self.envs = []
        self.actor_handles = []
        
        # Robot Asset (A simple cylinder representing a mobile base)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.disable_gravity = True # Robot doesn't fall
        self.robot_asset = self.gym.create_capsule(self.sim, 0.2, 0.5, asset_options)
        
        self._load_spatial_embeddings()
        self._create_envs()
        
        self.gym.prepare_sim(self.sim)
        
        # Tensor APIs
        from isaacgym import gymtorch
        _root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(_root_tensor)
        
       
        self.gym.refresh_actor_root_state_tensor(self.sim)
        
        # Slicing 1::2 because there are 2 actors per env (Mesh first, Robot second)
        self.robot_positions = self.root_states[1::2, 0:3]
        self.robot_velocities = self.root_states[1::2, 7:10]
        
        self.targets = torch.zeros((self.num_envs, 3), device=self.device)
        self.is_bedroom_tensor = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for i, scene in enumerate(self.scenes):
            self.is_bedroom_tensor[i] = self.scene_to_is_bedroom.get(scene, False)
            
        self.reset()
        
    def _load_spatial_embeddings(self):
        print("Loading Pre-Computed Spatial Embeddings...")
        self.scene_coords = []
        self.scene_labels = []
        self.scene_embeddings_orig = []
        self.scene_embeddings_robo = []
        self.scenes = []
        
        # Count how many of each we need (Balanced 50/50 split)
        target_bedrooms = self.num_envs // 2
        target_non_bedrooms = self.num_envs - target_bedrooms
        
        loaded_bedrooms = 0
        loaded_non_bedrooms = 0
        
        for scene in sorted(os.listdir(self.scans_dir)):
            if len(self.scenes) >= self.num_envs:
                break
                
            is_bed = self.scene_to_is_bedroom.get(scene, False)
            
            # Skip if we already have enough of this class
            if is_bed and loaded_bedrooms >= target_bedrooms:
                continue
            if not is_bed and loaded_non_bedrooms >= target_non_bedrooms:
                continue
                
            emb_path = os.path.join(self.scans_dir, scene, "spatial_embeddings.pt")
            if os.path.exists(emb_path):
                # Load to CPU first to prevent gymtorch CUDA context corruption
                data = torch.load(emb_path, map_location="cpu")
                
                # Verify tensor shapes
                if data["feat_orig"].shape[0] != data["coords"].shape[0]:
                    print(f"Skipping {scene} due to mismatched tensor sizes.")
                    continue
                
                self.scenes.append(scene)
                self.scene_embeddings_orig.append(data["feat_orig"].to(self.device))
                self.scene_embeddings_robo.append(data["feat_robo"].to(self.device))
                self.scene_coords.append(data["coords"].to(self.device))
                self.scene_labels.append(data["labels"].to(self.device))
                
                if is_bed:
                    loaded_bedrooms += 1
                else:
                    loaded_non_bedrooms += 1
            else:
                print(f"Warning: Missing spatial_embeddings.pt for {scene}")
                
    def _create_envs(self):
        print("Creating environments and loading meshes...", flush=True)
        spacing = 10.0
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        
        for i, scene in enumerate(self.scenes):
            print(f"  Creating env {i} for scene {scene}...", flush=True)
            env = self.gym.create_env(self.sim, lower, upper, int(np.sqrt(self.num_envs)))
            self.envs.append(env)
            
            # Load static mesh (URDF)
            urdf_path = os.path.join(scene, f"{scene}.urdf")
            asset_options = gymapi.AssetOptions()
            asset_options.fix_base_link = True
            try:
                mesh_asset = self.gym.load_asset(self.sim, self.scans_dir, urdf_path, asset_options)
                pose = gymapi.Transform()
                # collision filter = 1 to prevent collision with robot
                self.gym.create_actor(env, mesh_asset, pose, scene, i, 1)
                print(f"    -> Actor created for {scene}", flush=True)
            except Exception as e:
                print(f"    -> Failed to load mesh {scene}: {e}", flush=True)
                
            # Load Robot
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0, 0, 1.0)
            # collision filter = 1 to prevent collision with mesh
            robot_actor = self.gym.create_actor(env, self.robot_asset, pose, "robot", i, 1)
            self.actor_handles.append(robot_actor)
            
    def reset(self):
        # Instead of doing the math here, just call reset_idx for ALL environments
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        self.env_steps[:] = 0
        self.reset_idx(all_env_ids)
        return self.compute_observations()
        
    def reset_idx(self, env_ids):
        """Resets specific environments without interrupting the others."""
        if len(env_ids) == 0:
            return

        # 1. Reset targets for the specific environments (Privacy task)
        # (Targets are no longer needed for Privacy, as we use dynamic T-Maze zones)

        # 2. Reset robot positions dynamically to a safe floor spot
        for idx in env_ids:
            env_i = idx.item()
            
            # Get the coordinates and labels for this specific scene
            coords = self.scene_coords[env_i]
            labels = self.scene_labels[env_i]
            
            # Find all points classified as Floor (label == 1)
            floor_indices = (labels == 1).nonzero(as_tuple=True)[0]
            
            if len(floor_indices) > 0:
                # Pick a random floor point
                rand_idx = torch.randint(0, len(floor_indices), (1,)).item()
                safe_pos = coords[floor_indices[rand_idx]]
                
                # Set X, Y to the floor point, and Z slightly above it so the robot drops in
                self.root_states[1 + 2 * env_i, 0:2] = safe_pos[0:2]
                self.root_states[1 + 2 * env_i, 2] = safe_pos[2] + 0.5 
            else:
                # Fallback if no floor is detected
                self.root_states[1 + 2 * env_i, 0:3] = torch.tensor([0.0, 0.0, 1.0], device=self.device)
                
        # Record starting X position for T-Maze zones
        self.start_x[env_ids] = self.root_states[1 + 2 * env_ids, 0]

        # Zero out velocities
        robot_indices = 1 + 2 * env_ids
        self.root_states[robot_indices, 7:13] = 0.0

        # 3. Update physics state
        self.gym.set_actor_root_state_tensor(
            self.sim, 
            gymtorch.unwrap_tensor(self.root_states)
        )
        
        self.env_steps[env_ids] = 0
    def step(self, actions):
        self.env_steps += 1
        actions = actions.to(self.device)
        
       
        actions = torch.clamp(actions, -1.0, 1.0)
        
        self.root_states[1::2, 7:9] = actions * 2.0 # max 2m/s
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
        
        for _ in range(10): # 60Hz physics -> 6Hz decision making
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            
        self.gym.refresh_actor_root_state_tensor(self.sim)
        
        obs = self.compute_observations()
        rewards, dones = self.compute_rewards()
        
        env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
            # Recompute observations ONLY so PPO gets the correct fresh start state
            obs = self.compute_observations() 
            
        return obs, rewards, dones, {}
        
    def compute_observations(self):
        # Safety catch: if physics exploded (robot spawned inside furniture)
        if torch.isnan(self.robot_positions).any():
            nan_envs = torch.isnan(self.robot_positions[:, 0]).nonzero(as_tuple=True)[0]
            for env_i in nan_envs:
                self.root_states[1 + 2 * env_i, 0:3] = torch.tensor([0.0, 0.0, 1.0], device=self.device)
                self.root_states[1 + 2 * env_i, 7:13] = 0.0
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            
        pooled_emb = torch.zeros((self.num_envs, 512 if not self.use_roboshape else 64), device=self.device)
        
        for i, scene in enumerate(self.scenes):
            pos = self.robot_positions[i] # (3,)
            coords = self.scene_coords[i] # (N, 3)
            if self.use_roboshape:
                feat = self.scene_embeddings_robo[i] # (N, 64)
            else:
                feat = self.scene_embeddings_orig[i] # (N, 512)
            
            dists = torch.norm(coords - pos, dim=1)
            closest_idx = torch.argmin(dists)
            pooled_emb[i] = feat[closest_idx]
            
        self.obs_buf = torch.cat([
            pooled_emb,                 # (num_envs, embed_dim)
            self.robot_positions        # (num_envs, 3)
        ], dim=-1)
            
        return self.obs_buf
        
    def compute_rewards(self):
        rewards = torch.zeros(self.num_envs, device=self.device)
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        if self.privacy_task:
            # : T-Maze Zone Evaluation ---
            x_dist = self.robot_positions[:, 0] - self.start_x
            
            # Has it moved 1.0 meter left or right from its spawn?
            went_left = (x_dist < -1.0)
            went_right = (x_dist > 1.0)
            
            is_bed = self.is_bedroom_tensor
            
            # If Bedroom -> Go Left. If Non-Bedroom -> Go Right.
            correct = (is_bed & went_left) | (~is_bed & went_right)
            wrong = (is_bed & went_right) | (~is_bed & went_left)
            
            dones = correct | wrong
            
            rewards = torch.zeros_like(x_dist)
            rewards[correct] = 10.0
            rewards[wrong] = -10.0
            
            # Step penalty for wandering without committing
            rewards[~dones] = -0.1
        else:
            # Utility Task: Wall Seeking
            for i in range(self.num_envs):
                pos = self.robot_positions[i]
                coords = self.scene_coords[i]
                labels = self.scene_labels[i]
                
                # Reverting back to 3D distance so it matches the observation
                dists = torch.norm(coords - pos, dim=1)
                
                closest_idx = torch.argmin(dists)
                label = labels[closest_idx].item()
                
                if label == 0: # Wall
                    rewards[i] += 10.0
                    dones[i] = True
                elif label != 1: # Not Floor (i.e. Furniture)
                    rewards[i] -= 0.1
                else:
                    rewards[i] += 0.0 # Floor
                    
        # Apply timeout
        timeouts = (self.env_steps >= 200)
        dones = dones | timeouts
                    
        return rewards, dones
