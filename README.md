# SHGLNN
## Self-supervised Heterogeneous Hypergraph Learning with Context-aware Pooling for Graph-level Classification
SHGLNN is an algorithm for graph embeddings, harnessing the power of hypergraphs for intra and inter-graph contexts. This repository provides the code implementation of the algorithm as described in the paper.

## Algorithm Overview
The algorithm receives a set of graphs and processes them in batches through a specified number of epochs. It leverages intra-graph hyperedges, inter-graph hyperedges, type-specific node and hyperedge attentions, and contrastive loss to generate embeddings for the input graphs.

See Algorithm 1 in our paper for a detailed step-by-step explanation.

## Code Structure
### Core Files:
- **kMatrix.py**: Responsible for generating k-specific matrices corresponding to intra-graph hyperedges.
- **sMatrix.py**: Generates matrix pertaining to inter-graph hyperedges.
- **hypergraphGen.py**: Constructs hypergraph using the intra and inter-graph hyperedges.
- **dataset_loader.py**: Helper functions to load and preprocess the datasets.
- **layers.py**: Contains various neural network layers utilized in the model - node convolution, hyperedge convolution, context-aware graph-level pooling
- **model.py**: Defines the main SHGLNN neural network model.
- **train.py**: Orchestrates the training process of the algorithm, including the calculation of contrastive loss and back propagation.

## Prerequisites
- Python 3.8 or above
- PyTorch Geometric 2.4.0
- NumPy 1.26.0
- NetworkX 3.1
- HypernetX 2.0.5

## Datasets
We use the TUDatasets mentioned in the paper from PyTorch Geometric TUDatasets (https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.TUDataset.html#torch_geometric.datasets.TUDataset)
## Usage
1. Clone the repository
`git clone https://github.com/YOUR_USERNAME/SHGLNN.git`
`cd SHGLNN`
2. Install the required packages/libraries:
`pip install numpy networkx hypernetx`
3. Generate the matrices K and S for intra- and inter-graph hyperedges
   - Note that for all datasets, multiple K matrices would be generated as we go through different values of K. However, we will generate a single S matrix for each dataset.
5. Generate hypergraph using `hypergraphGen.py`
6. Develop layers using `layers.py`
7. Run the `model.py` to develop the model
8. Then, train the model using `train.py`
   - Note that for datasets, we used the built-in TU dataset from PyTorch Geometric

## Contribution
Pull requests are welcome. For major changes or any issues, please open an issue first for discussion. For further inquiries, contact the first author at malik.khizar@hdr.mq.edu.au or khizerhayat92@gmail.com.

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Citation
If you use this work, please cite the following:
```
@inproceedings{hayatSHGLNN2023, 
    author = {Hayat, Malik Khizar, Xue, Shan and Yang, Jian},
    title = {Self-supervised Heterogeneous Hypergraph Learning with Context-aware Pooling for Graph-level Classification},
    booktitle = {ICDM},
    year = {2023},
    pages=---,
    doi={---},
}
```
