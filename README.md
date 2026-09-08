# RoboShape
Information Theoretical Pipeline for Privacy Preserving Intelligent Robotics Sensing
<img width="2688" height="1476" alt="readmegraphic" src="https://github.com/user-attachments/assets/515d6f09-b255-43be-8c9b-bf4df2d31b74" />

### 🚀 Getting Started
#### 1. Installation
This repository provides a conda environment file to easily install all dependencies (including PyTorch, IsaacGym, and other utilities).
````bash
# Clone the repository
git clone https://github.com/TMLR26RS/RoboShape.git
cd RoboShape
# Create the conda environment from the provided yaml file
# Note: The file is named enviroment.yaml
conda env create -f enviroment.yaml

2. Project Structure
The codebase is modularized into data preparation, baseline classifiers, Isaac Gym reinforcement learning environments, and core model training.

RoboShape/
├── classifiers/               # Baseline classification scripts for privacy/utility evaluation
│   ├── classifier_noisy.py    # Baseline with noise injection
│   ├── classifier_random_encoder.py
│   └── eval_classifiers.py    # Evaluation logic for classifiers
    └── classifier2.py         # Classifier for comparing RoboShape embeddings vs original embeddings
├── data_prep/                 # Scripts to preprocess 3D scene datasets
│   ├── preprocess_hm3d.py     # HM3D dataset processing
│   ├── preprocess_scannet.py  # ScanNet dataset processing
│   ├── inference_manual.py    # Inference script for obtaining 512-dim Sonata embeddings
│   └── ...                    # Various histogram and feature extraction utilities
├── roboshape_isaacgym/        # RL navigation environment using Isaac Gym
│   ├── train_policy.py        # Main RL training script
│   ├── scannet_nav_env.py     # Custom ScanNet navigation environment
│   └── generate_scene_embeddings.py
├── src/                       # Core model architecture and utilities
│   ├── models/                # Encoder architectures and model definitions
│   ├── data/                  # Dataloaders and dataset abstractions
│   └── dual_optimization_encoder.py
├── enviroment.yaml            # Conda environment dependencies
├── train.py                   # Main training loop for RoboShape models
└── test.py                    # Inference and testing routines
└── roboshape.py               # main file

3. Reproducibility & Running Scripts
To reproduce the experiments, follow this pipeline:

Step A: Data Preparation Before training, you must preprocess the raw datasets (ScanNet, HM3D, etc.) and extract the required geometric features.

# Example: Preprocess ScanNet dataset
python data_prep/preprocess_scannet.py

# Run manual inference to obtain 512-dim Sonata embeddings
python data_prep/inference_manual.py

# Generate feature embeddings for baseline evaluations
python data_prep/save_encoder_features_matterport3d.py

Step B: Train the RoboShape Encoder To train the primary representation encoder with the dual optimization (utility vs privacy) objective:
````
# Feature Extractor: Sonata
We used PTv3 pre-trained model Sonata as feature extractor in order to use point cloud modalities.
You can find the model details here: https://github.com/facebookresearch/sonata

At the end of the feature extractor encoder layers, model supplies points with 512 dimensions. 

# Dataset: Scannet
This project utilizes the Scannet Dataset for 3D object detection. 

Visualizations:

Below are samples of the point clouds visualizations obtained by running sonata_ınference.py using Scannet Scenes:

<table width="100%">
  <tr>
    <td align="center" width="20%">
      <img width="100%" alt="fig1" src="https://github.com/user-attachments/assets/a098d26d-81db-4879-945b-59dc0bf264c4" />
      <br>
      <i>Figure 1</i>
    </td>
    <td align="center" width="20%">
      <img width="100%"  alt="fig2" src="https://github.com/user-attachments/assets/c467442a-6210-455d-b125-9cbc48cd4a94" />
      <br>
      <i>Figure 2</i>
    </td>
    <td align="center" width="20%">
      <img width="100%"  alt="fig3" src="https://github.com/user-attachments/assets/138e0134-d72a-462e-b2ad-81a9d7fea468" />
      <br>
      <i>Figure 3</i>
    </td>
    <td align="center" width="20%">
      <img width="100%" alt="fig4" src="https://github.com/user-attachments/assets/01021ecc-dccc-410b-9aba-8c2e24572d49" />
      <br>
      <i>Figure 4</i>
    </td>
    <td align="center" width="20%">
      <img width="100%" alt="fig5" src="https://github.com/user-attachments/assets/21e6f08a-635f-477c-9483-aa779deabd40" />
      <br>
      <i>Figure 5</i>
    </td>
  </tr>
</table>

The furniture distributions over scenes , and the comparison of ground-truth lables and the segmentation of the Sonata are as follows.

<div align="center">
<img width="2700" height="1050" alt="fig6" src="https://github.com/user-attachments/assets/845bb3b8-3eab-4a8b-b1a9-645923b8378f" />

</div>

<div align="center">
<img width="2700" height="750" alt="fig7" src="https://github.com/user-attachments/assets/fa77d326-bedd-4ec4-b26b-8371d89a20ad" />


</div>

Distribution of numbers of points over all scenes as follows:

<div align ="center">
<img width="1666" height="1038" alt="fig8" src="https://github.com/user-attachments/assets/492bbdf1-23ba-4960-a81d-9bd95e03d57e" />

</div>

Distributions of furnitures over scene types :
<div align= "center">
<img width="3578" height="1777" alt="fig9" src="https://github.com/user-attachments/assets/5607a29d-a2ec-4487-8c9b-c88bc94cfd66" />


</div>


Sonata Encoder unites different points during downsampling the raw points in the encoding process, find the number of different furniture lables for each voxel for scannet dataset below:

<div align= "center">
<img width="1786" height="884" alt="fig10" src="https://github.com/user-attachments/assets/05c1cc9e-e050-4065-a5a7-b7f1aa9d7bd5" />

</div>

Download & Setup:

Please request access to the ScanNet dataset and download it from the official ScanNet Benchmark: http://www.scan-net.org/.
# Training:

# Scannet
<div align= "center">
<img width="2117" height="865" alt="fig11" src="https://github.com/user-attachments/assets/4af5578d-05bf-437b-be99-1fcbd1572df9" />

</div>

# Matterport3D

<div align= "center">
<img width="1350" height="900" alt="fig12" src="https://github.com/user-attachments/assets/5a9c418c-af7a-4c9a-bf36-2b032b8272fb" />


</div>
<div align= "center">
 <img width="2117" height="865" alt="fig13" src="https://github.com/user-attachments/assets/decb4c7a-6e6a-41af-b090-6b39fc1922df" />



</div>

# ARKitScenes

<div align= "center">
<img width="1350" height="900" alt="fig14" src="https://github.com/user-attachments/assets/2dc970fe-0baa-4679-8b8c-ff8b8387bd43" />

</div>
<div align= "center">
<img width="2117" height="865" alt="fig15" src="https://github.com/user-attachments/assets/c530054b-2944-4e96-a318-710190ceb84e" />




</div>



# Results: 
# Scannet 
4 different classifiers trained in order to show the success of roboshape embeddings at hiding private attributes. 2 of them trained in order to classify sonata embeddings according to furniture type (public label) and scenetype ( private label), and the other 2 of them in order to classify roboshape embeddings according to public and private labels. You can find the Train , test losses and classifying accuracies below.
<div align= "center">
<img width="1271" height="359" alt="fig16" src="https://github.com/user-attachments/assets/19f2eb94-9871-47f4-9e2c-e9c68de06a30" />

</div>

<div align= "center">

<img width="1275" height="366" alt="fig17" src="https://github.com/user-attachments/assets/e2ebab65-65e4-4340-9cdf-0c9874c72ede" />


</div>

Loss curves of classifiers for noisy original embeddings:
<div align= "center">
<img width="1800" height="600" alt="fig18" src="https://github.com/user-attachments/assets/505ef196-7500-4104-aefd-d3192c000215" />


</div>

Loss curves of classifiers for randomly initialized encoder outputs:

<div align= "center">
<img width="1800" height="600" alt="fig19" src="https://github.com/user-attachments/assets/4fddfec5-a9ab-4138-988e-8d368ad912a9" />

</div>


Auroc results for 4 different baselines. 
<div align= "center">
  <img width="1272" height="532" alt="fig20" src="https://github.com/user-attachments/assets/bb18ffa2-3a22-4076-a2a2-3c62be951146" />

</div>

# Matterport
4 different classifiers trained in order to show the success of roboshape embeddings at hiding private attributes. 2 of them trained in order to classify sonata embeddings according to furniture type (public label) and scenetype ( private label), and the other 2 of them in order to classify roboshape embeddings according to public and private labels. You can find the Train , test losses and classifying accuracies below.
<div align= "center">
<img width="2100" height="600" alt="fig21" src="https://github.com/user-attachments/assets/fa466030-b78e-496c-bb97-80335143afa5" />


</div>

<div align= "center">

<img width="2100" height="600" alt="fig22" src="https://github.com/user-attachments/assets/4281e746-4492-4fa1-976e-91d432796c54" />


</div>

Loss curves of classifiers for noisy original embeddings:
<div align= "center">
<img width="1800" height="600" alt="fig23" src="https://github.com/user-attachments/assets/0a015f2d-ca83-4581-9ef3-dff1233be571" />



</div>

Loss curves of classifiers for randomly initialized encoder outputs:

<div align= "center">
<img width="1800" height="600" alt="fig24" src="https://github.com/user-attachments/assets/982f0a1b-46cc-4ebe-8919-db9e6a8332ef" />


</div>
Auroc results for 4 different baselines:
<div align= "center">
<img width="1800" height="750" alt="auroc_new" src="https://github.com/user-attachments/assets/9c6186fd-8ef4-4317-8e5e-2df66621d483" />




</div>

# ARKitScenes

4 different classifiers trained in order to show the success of roboshape embeddings at hiding private attributes. 2 of them trained in order to classify sonata embeddings according to furniture type (public label) and scenetype ( private label), and the other 2 of them in order to classify roboshape embeddings according to public and private labels. You can find the Train , test losses and classifying accuracies below.
<div align= "center">
<img width="2100" height="600" alt="fig26" src="https://github.com/user-attachments/assets/eed898b7-ed57-44f4-9a55-32be642dcc3b" />


</div>

<div align= "center">

<img width="1238" height="338" alt="fig27" src="https://github.com/user-attachments/assets/027d8e5f-d36d-4b34-9a90-410c464e915c" />



</div>

Loss curves of classifiers for noisy original embeddings:
<div align= "center">
<img width="1237" height="397" alt="fig28" src="https://github.com/user-attachments/assets/4ad161f8-4ed7-4129-b30b-c00495d57695" />




</div>

Loss curves of classifiers for randomly initialized encoder outputs:

<div align= "center">
<img width="1236" height="390" alt="fig29" src="https://github.com/user-attachments/assets/41b23a7b-36db-4c79-9fbb-4a5a601dc69e" />

</div>


Auroc results for 4 different baselines. 
<div align= "center">
<img width="1237" height="520" alt="fig30" src="https://github.com/user-attachments/assets/231fd83c-1188-4e50-b68f-b37344f51a87" />


</div>
