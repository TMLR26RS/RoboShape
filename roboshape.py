import json
import logging
import math
from pathlib import Path
from typing import Tuple

import hydra
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Third party imports
import torch
import torch.nn as nn
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, TensorDataset

from src.data.utils import TexShapeDataset
from src.dual_optimization_encoder import DualOptimizationEncoder
from src.mine import Mine, MutualInformationEstimator
from src.models import models_to_train

# Local imports
from src.models.models_to_train import Encoder, MI_CalculatorModel
from src.models.utils import create_encoder_model, create_mi_calculator_model
from src.utils.config import (
    EncoderParams,
    ExperimentParams,
    MineParams,
    load_experiment_params,
)
from src.utils.general import configure_torch_backend, set_seed


@hydra.main(config_path="configs", config_name="config.yaml", version_base="1.2")
def main(config: DictConfig) -> None:
    """Main function to train the encoder model using dual optimization."""
    logging.basicConfig(level=logging.INFO)
    # Set random seed for reproducibility
    seed = config.seed
    set_seed(seed)
    configure_torch_backend()
    logging.info("Seed: %s", seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment_params: ExperimentParams = load_experiment_params(config)

    # Initialize encoder model
    encoder_model: Encoder = create_encoder_model(
        model_name=experiment_params.encoder_params.encoder_model_name,
        model_params=experiment_params.encoder_params.encoder_model_params,
    )
    experiment_dir_path: Path = Path(HydraConfig.get().runtime.output_dir)
    print(str(experiment_dir_path))
    encoder_model.to(device)
    print(encoder_model)

    # --- Resume from checkpoint if specified ---
    start_epoch: int = 0
    checkpoint_path = config.dataset.get("checkpoint_path", None)
    if checkpoint_path is not None:
        logging.info("Resuming from checkpoint: %s", checkpoint_path)
        encoder_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        start_epoch = int(config.dataset.get("start_epoch", 0))
        logging.info("Starting from epoch: %s", start_epoch)

    # Load the dataset
    # Create random data
    embeddings = torch.tensor(
        np.load(config.dataset.train_embeddings_path), dtype=torch.float32
    )
    label2 = torch.tensor(
        np.load(config.dataset.train_label1_path), dtype=torch.float32
    )
    label1 = torch.tensor(
        np.load(config.dataset.train_label2_path), dtype=torch.float32
    )
    print(
        embeddings.shape,
        label1.shape,
        label2.shape,
    )
    # Convert to numpy for splitting
    emb_np = embeddings.numpy()
    l1_np = label1.numpy()
    l2_np = label2.numpy()
    
    idx = np.arange(len(l1_np))
    idx_train, idx_temp = train_test_split(idx, test_size=0.20, random_state=42, stratify=l1_np)
    idx_val, idx_test = train_test_split(idx_temp, test_size=0.50, random_state=42, stratify=l1_np[idx_temp])

    _dataset = TexShapeDataset(
        embeddings=torch.from_numpy(emb_np[idx_train]).float(), 
        label1=torch.from_numpy(l1_np[idx_train]).float(), 
        label2=torch.from_numpy(l2_np[idx_train]).float()
    )
    _val_dataset = TexShapeDataset(
        embeddings=torch.from_numpy(emb_np[idx_val]).float(), 
        label1=torch.from_numpy(l1_np[idx_val]).float(), 
        label2=torch.from_numpy(l2_np[idx_val]).float()
    )
    logging.info("Train set: %s, Val set: %s, Test set: %s", len(_dataset), len(_val_dataset), len(idx_test))

    # Set the mine batch size
    mine_batch_size = experiment_params.mine_params.mine_batch_size
    if mine_batch_size == -1:
        mine_batch_size = len(_dataset)

    # Create a dataloader
    data_loader = DataLoader(
        _dataset,
        batch_size=mine_batch_size,
        shuffle=True,
        num_workers=32,
        pin_memory=True,
    )

    dual_optimization = DualOptimizationEncoder(
        encoder_model=encoder_model,
        data_loader=data_loader,
        val_dataset=_val_dataset,
        device=device,
        experiment_dir_path=Path(experiment_dir_path),
        mine_params=experiment_params.mine_params,
        encoder_params=experiment_params.encoder_params,
        beta=experiment_params.beta,
    )

    num_batches_final_mi = math.ceil(len(_dataset) / mine_batch_size)
    logging.info("Num batches Final MI: %s", num_batches_final_mi)

    # Train the encoder
    dual_optimization.train_encoder(
        num_batches_final_MI=num_batches_final_mi,
        include_privacy=True,
        include_utility=True,
        gradient_batch_size=1,
        start_epoch=start_epoch,
    )

    # Save encoder weights
    encoder_save_path = experiment_dir_path / "encoder_weights"
    encoder_save_path.mkdir(parents=True, exist_ok=True)


class DualOptimizationEncoder(nn.Module):
    """
    Dual Optimization Encoder Model. This model trains the encoder using dual optimization.
    """

    def __init__(
        self,
        *,
        encoder_model: models_to_train.Encoder,
        data_loader: DataLoader,
        val_dataset: TexShapeDataset,
        device: torch.device,
        experiment_dir_path: Path,
        mine_params: MineParams,
        encoder_params: EncoderParams,
        beta: float,
        **kwargs,
    ) -> None:
        super().__init__()
        self.encoder_model = encoder_model

        self.data_loader: DataLoader = data_loader
        self.dataset: TexShapeDataset = data_loader.dataset
        self.val_dataset: TexShapeDataset = val_dataset

        self.mine_params: MineParams = mine_params

        # Set the mine batch size if -1 passed
        if self.mine_params.mine_batch_size == -1:
            self.mine_params.mine_batch_size = len(self.dataset)

        self.encoder_params: EncoderParams = encoder_params
        self.experiment_dir_path: Path = experiment_dir_path

        self.encoder_learning_rate = encoder_params.encoder_learning_rate

        self.encoder_optimizer = torch.optim.Adam(
            self.encoder_model.parameters(),
            lr=self.encoder_learning_rate,
        )
        self.beta: float = beta
        # Define the device
        self.device: torch.device = device

        if kwargs.get("device_idx", None) is not None:
            self.device_idx: int = kwargs.get("device_idx", None)
        else:
            self.device_idx: int = 0

        self.num_workers: int = 0  # experiment_params.num_workers

        logging.info("Device: %s", self.device)

        self.epoch = 0

    def get_MINE(
        self,
        *,
        stats_network: nn.Module,
        device: torch.device,
    ) -> MutualInformationEstimator:
        # Define Mine model
        mi_estimator = Mine(stats_network, loss="mine").to(self.device)

        kwargs = {
            "mine": mi_estimator,
            "lr": 1e-3,
            "alpha": 0.1,  # Used as the ema weight in MINE
            # Determines how many mini batches (MINE iters) of gradients get accumulated before optimizer step gets applied
            # Meant to stabilize the MINE curve for [hopefully] better encoder training performance
        }

        model = MutualInformationEstimator(
            loss="mine",
            **kwargs,
        ).to(device)
        return model

    def get_transformed_data_and_loaders(
        self,
    ) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
        # Get encoder transformed data
        transformed_embeddings: torch.Tensor = self.encoder_model(
            self.dataset.embeddings.float().to(self.device)
        )

        labels_public = self.dataset.label1.float()
        labels_private = self.dataset.label2.float()

        # Define datasets for MINE

        # Public Label
        z_train_utility_detached = TensorDataset(
            transformed_embeddings.detach().cpu(),
            labels_public.detach(),
        )

        # Private Label
        z_train_privacy_detached = TensorDataset(
            transformed_embeddings.detach().cpu(),
            labels_private.detach(),
        )

        # Define dataloaders for MINE
        z_train_loader_utility_detached = DataLoader(
            z_train_utility_detached,
            self.mine_params.mine_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            # TODO: Check here
            # pin_memory=True,
        )
        z_train_loader_privacy_detached = DataLoader(
            z_train_privacy_detached,
            self.mine_params.mine_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            # TODO: Check here
            # pin_memory=True,
        )

        return (
            z_train_loader_utility_detached,
            z_train_loader_privacy_detached,
            transformed_embeddings,
        )

    def forward(
        self,
        *,
        num_batches_final_MI: int,
        include_privacy: bool = True,
        include_utility: bool = True,
        gradient_batch_size: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, MutualInformationEstimator, MutualInformationEstimator]:
        
        """
        Forward pass for the dual optimization model

        Args:
            num_batches_final_MI (int): Number of batches to calculate the final MI estimate
            include_privacy (bool, optional): Include privacy in the training. Defaults to True.
            include_utility (bool, optional): Include utility in the training. Defaults to True.
            gradient_batch_size (int, optional): . Defaults to 1.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: MI estimates for utility and privacy
        """

        transformed_embeddings: torch.Tensor

        (
            z_train_loader_utility_detached,
            z_train_loader_privacy_detached,
            transformed_embeddings,
        ) = self.get_transformed_data_and_loaders()

        # TODO: Fix this part
        # Get MINE model (sitting in Pytorch lightning module)
        model_MINE_utility: MutualInformationEstimator
        model_MINE_privacy: MutualInformationEstimator

        MINE_utility_stats_network: nn.Module = self._get_utility_stats_network()
        MINE_privacy_stats_network: nn.Module = self._get_privacy_stats_network()

        model_MINE_utility = self.get_MINE(
            stats_network=MINE_utility_stats_network,
            device=self.device,
        )

        model_MINE_privacy = self.get_MINE(
            stats_network=MINE_privacy_stats_network,
            device=self.device,
        )

       
        last_mi_utility = torch.tensor(0)
        last_mi_privacy = torch.tensor(0)
        if include_utility:
            early_stop_callback = EarlyStopping(
                monitor="mi",
                patience=self.mine_params.mine_trainer_patience,
                mode="max",
            )

            logger_utility = TensorBoardLogger(
                str(self.experiment_dir_path),
                name="MINE_logs",
                version=f"utility_{self.epoch}",
            )

            trainer_utility = Trainer(
                max_epochs=self.mine_params.mine_epochs_utility,
                logger=logger_utility,
                log_every_n_steps=1,
                accelerator="gpu",
                devices=[self.device_idx],
                accumulate_grad_batches=gradient_batch_size,
                callbacks=[early_stop_callback],
            )

            trainer_utility.fit(
                model=model_MINE_utility,
                train_dataloaders=z_train_loader_utility_detached,
            )

           

            labels_public = self.dataset.label1.float()
            z_train_utility = TensorDataset(
                transformed_embeddings, labels_public.float()
            )

           

            z_train_loader_utility = DataLoader(
                z_train_utility,
                batch_size=self.mine_params.mine_batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )
            model_MINE_utility.energy_loss.to(self.device)
            sum_MI_utility = 0

            logging.info(
                "Num batches final MI: %s, \tLen dataset: %s, \tK: %s, \tLen dataset / K: %s",
                num_batches_final_MI,
                len(z_train_loader_utility.dataset),
                self.mine_params.mine_batch_size,
                len(z_train_loader_utility.dataset) / self.mine_params.mine_batch_size,
            )

            assert num_batches_final_MI <= (
                math.ceil(
                    len(z_train_loader_utility.dataset)
                    / self.mine_params.mine_batch_size
                )
            )
            utility_it = iter(z_train_loader_utility)
            for _ in range(num_batches_final_MI):
                Tx, Lx = next(utility_it)
                Tx, Lx = Tx.to(self.device), Lx.to(self.device)
                sum_MI_utility += model_MINE_utility.energy_loss(Tx, Lx)

            # MINE loss = -1 * MI estimate since we are maximizing using gradient descent still
            last_mi_utility: torch.Tensor = -1 * sum_MI_utility / num_batches_final_MI

        if include_privacy:
            early_stop_callback = EarlyStopping(
                monitor="mi",
                patience=self.mine_params.mine_trainer_patience,
                mode="max",
            )

            logger_privacy = TensorBoardLogger(
                str(self.experiment_dir_path),
                name="MINE_logs",
                version=f"privacy_{self.epoch}",
            )

            trainer = Trainer(
                max_epochs=self.mine_params.mine_epochs_privacy,
                logger=logger_privacy,
                log_every_n_steps=1,
                accelerator="gpu",
                devices=[self.device_idx],
                accumulate_grad_batches=gradient_batch_size,
                callbacks=[early_stop_callback],
            )

            trainer.fit(
                model=model_MINE_privacy,
                train_dataloaders=z_train_loader_privacy_detached,
            )

            # # If path exists, save the weights of the MINE model
            # if self.privacy_stats_network_path:
            #     logging.info("Using MI Strategy, saving MINE model")
            #     torch.save(
            #         model_MINE_privacy.energy_loss.state_dict(),
            #         self.privacy_stats_network_path,
            #     )

            #     ## -------- Calculate I(T(x); S(x)) estimate after MINE training ---------- ##
            labels_private = self.dataset.label2.float()
            z_train_privacy = TensorDataset(transformed_embeddings, labels_private)
            z_train_loader_privacy = DataLoader(
                z_train_privacy,
                self.mine_params.mine_batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )
            model_MINE_privacy.energy_loss.to(self.device)

            assert num_batches_final_MI <= math.ceil(
                len(z_train_loader_privacy.dataset) / self.mine_params.mine_batch_size
            )
            sum_MI_privacy = 0
            privacy_it = iter(z_train_loader_privacy)
            # prev_mi = None
            for _ in range(num_batches_final_MI):
                Tx: torch.Tensor
                Sx: torch.Tensor
                Tx, Sx = next(privacy_it)
                Tx, Sx = Tx.to(self.device), Sx.to(self.device)
                sum_MI_privacy += model_MINE_privacy.energy_loss(Tx, Sx)

            last_mi_privacy: torch.Tensor = -1 * sum_MI_privacy / num_batches_final_MI

        logging.info(
            "Final MI values: utility: %s, privacy: %s",
            last_mi_utility.item(),
            last_mi_privacy.item(),
        )
        return last_mi_utility, last_mi_privacy, model_MINE_utility, model_MINE_privacy

    def train_encoder(
        self,
        *,
        num_batches_final_MI: int = 100,
        include_privacy: bool = True,
        include_utility: bool = True,
        gradient_batch_size: int = 1,
        start_epoch: int = 0,
    ) -> None:
        """
        Training function for the encoder model.

        Args:
            num_batches_final_MI (int, optional): Number of batches to calculate the final MI estimate. Defaults to 100.
            include_privacy (bool, optional): Include privacy in the training. Defaults to True.
            include_utility (bool, optional): Include utility in the training. Defaults to True.
            gradient_batch_size (int, optional): . Defaults to 1.
            start_epoch (int, optional): Epoch to resume from. Epochs before this are skipped. Defaults to 0.

        Returns:
            None
        """
        """K = MINE_BATCH_SIZE"""
        # Encoder's training params
        mi_utility: torch.Tensor
        mi_privacy: torch.Tensor
        self.encoder_model.train()
        self.epoch = start_epoch  # sync internal epoch counter with resume point

        # --- Load existing results.json if resuming, so we append rather than overwrite ---
        results_json_path = self.experiment_dir_path / "results.json"
        if results_json_path.exists():
            with open(results_json_path, "r") as f:
                epoch_results: list = json.load(f)
        else:
            epoch_results: list = []

        for epoch in range(start_epoch, self.encoder_params.num_enc_epochs):
            (
                mi_utility,
                mi_privacy,
                model_MINE_utility,
                model_MINE_privacy,
            ) = self.forward(
                num_batches_final_MI=num_batches_final_MI,
                include_privacy=include_privacy,
                include_utility=include_utility,
                gradient_batch_size=gradient_batch_size,
            )
            self.encoder_optimizer.zero_grad()

            # Calculate the score
            loss: torch.Tensor = -mi_utility + self.beta * mi_privacy
            loss.backward()
            self.encoder_optimizer.step()

            # --- Save encoder checkpoint & MINE Weights ---
            enc_save_dir = self.experiment_dir_path / "encoder_weights"
            if not enc_save_dir.exists():
                enc_save_dir.mkdir(parents=True, exist_ok=True)
            enc_save_path = enc_save_dir / f"model_{epoch}.pt"
            self._save_encoder_weights(enc_save_path)

            mine_util_path = enc_save_dir / f"mine_utility_epoch={epoch}.pt"
            mine_priv_path = enc_save_dir / f"mine_privacy_epoch={epoch}.pt"
            torch.save(model_MINE_utility.energy_loss.state_dict(), mine_util_path)
            torch.save(model_MINE_privacy.energy_loss.state_dict(), mine_priv_path)

            # --- Evaluate on Validation Set ---
            self.encoder_model.eval()
            model_MINE_utility.energy_loss.eval()
            model_MINE_privacy.energy_loss.eval()

            with torch.no_grad():
                # Process in chunks if validation set is too large for single forward pass, 
                # but usually 16k embeddings fits in memory easily.
                val_emb = self.val_dataset.embeddings.float().to(self.device)
                Tx_val = self.encoder_model(val_emb)
                
                Lx_val = self.val_dataset.label1.float().to(self.device)
                Sx_val = self.val_dataset.label2.float().to(self.device)

                val_mi_utility = -1 * model_MINE_utility.energy_loss(Tx_val, Lx_val)
                val_mi_privacy = -1 * model_MINE_privacy.energy_loss(Tx_val, Sx_val)
                val_loss = -val_mi_utility + self.beta * val_mi_privacy
            
            self.encoder_model.train()

            # --- Persist per-epoch metrics to results.json ---
            epoch_record = {
                "epoch": epoch,
                "mi_utility": round(mi_utility.item(), 8),
                "mi_privacy": round(mi_privacy.item(), 8),
                "loss": round(loss.item(), 8),
                "val_mi_utility": round(val_mi_utility.item(), 8),
                "val_mi_privacy": round(val_mi_privacy.item(), 8),
                "val_loss": round(val_loss.item(), 8),
            }
            epoch_results.append(epoch_record)
            with open(results_json_path, "w") as f:
                json.dump(epoch_results, f, indent=2)
            
            self._plot_loss_curves(epoch_results)

            logging.info(
                "====> Epoch: %s Utility MI I(T(x); L(x)): %s",
                epoch,
                round(mi_utility.item(), 8),
            )
            logging.info(
                "====> Epoch: %s Privacy MI I(T(x); S(x)): %s",
                epoch,
                round(mi_privacy.item(), 8),
            )
            logging.info("====> Epoch: %s Loss: %s", epoch, round(loss.item(), 8))

            self.epoch += 1

    def _plot_loss_curves(self, results: list) -> None:
        epochs = [r["epoch"] for r in results]
        train_loss = [r["loss"] for r in results]
        val_loss = [r.get("val_loss", 0) for r in results]

        plt.figure(figsize=(8, 6))
        plt.plot(epochs, train_loss, marker="o", label="Train Loss")
        if val_loss and val_loss[0] != 0:
            plt.plot(epochs, val_loss, marker="s", label="Val Loss")
        
        plt.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (-utility + beta*privacy)")
        plt.title("Train and Validation Loss")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        
        plot_path = self.experiment_dir_path / "train_val_loss_curve.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()

    def _get_utility_stats_network(self) -> MI_CalculatorModel:
        stats_network = create_mi_calculator_model(
            model_name=self.mine_params.utility_stats_network_model.model_name,
            model_params=self.mine_params.utility_stats_network_model.model_params,
        )
        return stats_network

    def _get_privacy_stats_network(self) -> MI_CalculatorModel:
        stats_network = create_mi_calculator_model(
            model_name=self.mine_params.privacy_stats_network_model.model_name,
            model_params=self.mine_params.privacy_stats_network_model.model_params,
        )
        return stats_network

    def _save_encoder_weights(self, model_path: Path) -> None:
        
        save_dir = model_path.parent

        logging.debug("Saving weights to %s", save_dir)

        # Save the state dict of the model
        torch.save(
            self.encoder_model.state_dict(),
            model_path,
        )

        optimizer_path = (
            save_dir
            / f"optimizer_{model_path.stem}.pt"
        )
        torch.save(
            self.encoder_optimizer.state_dict(),
            optimizer_path,
        )

    def save_mi_scores(self, epoch: int) -> None:
        raise NotImplementedError


if __name__ == "__main__":
    main()
