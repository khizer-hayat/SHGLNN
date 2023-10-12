# SHGLNN
## Self-supervised Heterogeneous Hypergraph Learning with Context-aware Pooling for Graph-level Classification

## Overview
- **kMatrix.py:** K matrix generation for k-hop neighborhood for intra-graph hyperedges.
- **sMatrix.py:** S matrix generation for inter-graph hyperedges.
- **hyperGen.py:** Hypergraph generation using K and S matrices.
- **layers.py:** Includes the nodeConv, hyperConv, and pooling layers of the model.
- **model.py:** the model SHGLNN.
- **train.py:** Model training.

- # SHGLNN Hypergraph Learning Model

This repository contains the implementation of the SHGLNN model which operates on hypergraphs for learning and analysis.

## Getting Started:

### Prerequisites:

Ensure that all required libraries are installed:

```bash
pip install torch numpy hypernetx


.
+-- hypergraphGen.py
+-- train.py
+-- model.py
+-- hyperedge_analysis.py
+-- dataset_preprocess.py
+-- hyperparameter_analysis.py
+-- data/
|   +-- MUTAG/
|   +-- PROTEINS/
|   +-- IMDB-BINARY/
|   +-- ... [other datasets]

# SHGLNN Hypergraph Learning Model

This repository contains the implementation of the SHGLNN model which operates on hypergraphs for learning and analysis.

## Getting Started

### Prerequisites

Before you begin, ensure that all required libraries are installed, which include `torch`, `numpy`, and `hypernetx`.

### Directory Structure

Your directory should be structured with the following main files:

- `hypergraphGen.py`: This script processes datasets and generates hypergraphs based on the K and S matrices.
  
- `train.py`: Used for model training and performance evaluation.

- `model.py`: Contains the model architecture of the SHGLNN.

- `hyperedge_analysis.py`: Used for hyperedge analysis.

- `hyperparameter_analysis.py`: Used for analyzing the effects of hyperparameters.

### Dataset Preprocessing

To preprocess a dataset and generate hypergraphs:

1. Ensure that your dataset's `kMatrix` and `sMatrix` are in the same directory.
2. Run `hypergraphGen.py` to process the dataset and generate hypergraphs.

### Model Training and Evaluation

To train and evaluate the model:

1. Ensure your hypergraphs are correctly generated.
2. Run `train.py` to train the model and evaluate its performance on the dataset.

### Hyperedge Analysis

To perform hyperedge analysis:

1. Ensure your hypergraphs are correctly generated.
2. Run `hyperedge_analysis.py`.

### Hyperparameter Analysis

For hyperparameter analysis on the COLLAB dataset:

1. Ensure your hypergraphs are correctly generated.
2. Run `hyperparameter_analysis.py`.

## Acknowledgments

Special thanks to the OpenAI community for guidance and support in creating this repository.
