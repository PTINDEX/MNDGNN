# MNDGNN

## Multiplex networks-based directed graph neural network for cancer driver gene identification

Identifying cancer driver genes is crucial in precision oncology. Most existing methods rely on a single interaction network to capture gene relationships. However, with the increasing availability of multi-omics and biological network data, integrating multiplex networks offers a more comprehensive representation of the complex and directional regulatory interactions among genes. Moreover, the number of validated cancer driver genes remains small compared with the vast number of unlabeled genes, leading to label scarcity and class imbalance. To address these limitations, we propose a multiplex networks-based directed graph neural network (MNDGNN). The model learns gene representations on multiplex networks with multi-omics data through directed graph convolution, which integrates neighbor diversity and degree diversity. We also incorporate data augmentation combining positive-sample augmentation with negative-sample inference to mitigate label scarcity. Experimental results show that the proposed method achieves better predictive performance and robustness than existing state-of-the-art methods. The predicted cancer driver genes are significantly enriched in cancer-related pathways and exhibit extensive interactions with known cancer driver genes, offering a new perspective for cancer driver gene discovery and the design of therapeutic strategies.

## Overview

### Data

`datasets/Feature_for_PyG.csv`: Multi-omics data

`datasets/PPI_PyG.txt`: PPI network

`datasets/Complexes_PyG.txt`: Protein complexes network

`datasets/KEGG_PyG.txt`: KEGG pathway network

`datasets/RegNetwork_PyG.txt`: RegNetwork

`datasets/DawnNet_PyG.txt`: DawnNet

`datasets/Kinase_Substrate_PyG.txt`: Kinase-substrate network

`datasets/kcdg_intersec_EID.csv`: Positive sample Entrez IDs

`datasets/mapping_dict_EID_Index.pickle`: Index-Entrez ID mapping

`datasets/maybe_cdg.txt`: Negative filtering gene set

`datasets/true_ids.txt`: Positive sample indices

### Utils

`utils/dataset.py`: Dataset interface

`utils/data_utils.py`: Graph preprocessing utilities

`utils/data_loading.py`: Data loading utilities

`utils/augmentation_util.py`: Data augmentation utilities

`utils/data_augmentation.py`: Data augmentation pipeline

`utils/predictions_utils.py`: Cancer driver gene prediction utilities

### Model

`model.py`: Model architecture

`main.py`: Training and evaluation script

`Best_hyperparams.yml`: Hyperparameter configuration

## Requirements

* Python 3.9.19

* PyTorch 2.1.2+cu118

* Pytorch Geometric 2.0.4

* Pytorch lightning 2.5.1

* Pytorch Sparse 0.6.18

* Deepod 0.4.1

* sklearn 0.24.2

* numpy 1.20.3

* yaml 0.2.5

## Usage

Run python main.py

## Citation

If you use this libarary in your work, please cite the paper:

