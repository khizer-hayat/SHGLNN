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

## Usage
1. Clone the repository
`git clone https://github.com/YOUR_USERNAME/SHGLNN.git`
`cd SHGLNN`
2. Install the required packages:
`pip install -r requirements.txt`
3. Train the model
`python train.py`

## Note
Ensure your dataset is structured correctly for dataset_loader.py. For more specific instructions, see the comments in the file.

## Contribution
Pull requests are welcome. For major changes, or any issue, please open an issue first to discuss what you would like to discuss.

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Citation

```
    @inproceedings{liu2022DAGAD, 
	    author = {Liu, Fanzhen and Ma, Xiaoxiao and Wu, Jia and Yang, Jian and Xue, Shan and Beheshti, Amin and  
                      Zhou, Chuan and Peng, Hao and Sheng, Quan Z. and Aggarwal, Charu C.},
	    title = {DAGAD: Data augmentation for graph anomaly detection},
	    booktitle = {ICDM},
	    year = {2022},
	    pages={259-268},
	    doi={10.1109/ICDM54844.2022.00036},
    }
```
