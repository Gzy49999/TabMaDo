# TabMaDo
This repository contains the official implementation code and supplementary materials for the paper: "TabMaDo: Dual-Guided Tabulamba Diffusion Oversampling for Imbalanced Tabular Data".

The structure and description of the files are as follows:

- `raw-dataset/`: This folder contains the raw datasets used in the experiments.
- `data_utils.py`: This module is responsible for data preprocessing, including data loading and feature engineering.
- `Tabulamba.py`: As the core module of the project, it implements the complete training process of the diffusion model. It includes the model definitions for the Diffuser and the lightweight Guider.
- `Supplementary_materials/`: This folder provides the supplementary materials for the paper, including the theoretical proof and additional detailed experimental results not fully presented in the main paper.
