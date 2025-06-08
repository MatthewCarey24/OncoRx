# OncoRx: Cancer Drug Response Prediction using Latent Factor Models

OncoRx is a machine learning framework for predicting cancer drug response using latent factor models. The project combines matrix factorization techniques with neural networks to predict IC50 values for drug-cell line combinations, with applications to both cell line data and patient samples.

## Overview

This project implements a novel approach to cancer drug response prediction by:

1. **Latent Factor Modeling**: Using Truncated SVD to decompose cell line × drug response matrices into latent representations
2. **Neural Network Prediction**: Training TensorFlow models on concatenated latent vectors to predict IC50 values
3. **Hyperparameter Optimization**: Using Optuna for automated hyperparameter tuning
4. **Pharmacogenomic Mapping**: Bridging gene expression data to pharmacogenomic space for patient applications
5. **Comprehensive Evaluation**: Using multiple metrics including Spearman correlation and NDCG

## Project Structure

```
OncoRx/
├── README.md                    # This file
├── environment.yml              # Conda environment specification
├── latent_model.py             # Main optimization script with Optuna
├── my_model.ipynb              # Jupyter notebook with full analysis pipeline
├── cadrres_sc/                 # Core package modules
│   ├── evaluation.py           # Evaluation metrics (Spearman, NDCG)
│   ├── utility.py              # Utility functions for analysis
│   └── pp/                     # Preprocessing modules
│       ├── gexp.py             # Gene expression preprocessing
│       └── scgexp.py           # Single-cell gene expression preprocessing
├── data/                       # Data directory
│   ├── essential_genes.txt     # List of essential genes
│   ├── IntOGen-DriverGenes.tsv # Cancer driver genes
│   ├── GDSC/                   # GDSC dataset files
│   │   ├── gdsc_all_abs_ic50_bayesian_sigmoid_only9dosages.csv
│   │   ├── GDSC_exp.tsv        # Gene expression data
│   │   └── GDSC_tissue_info.csv # Tissue information
│   └── patient/                # Patient data
│       ├── log2_fc_cluster_tpm.csv
│       └── percent_patient_tpm_cluster.xlsx
└── preprocessed_data/          # Preprocessed datasets
    └── GDSC/
        ├── drug_stat.csv       # Drug statistics
        └── hn_drug_stat.csv    # Head & neck drug statistics
```

## Key Features

### 1. Latent Factor Model
- **Matrix Factorization**: Uses Truncated SVD to decompose cell line × drug response matrices
- **Feature Engineering**: Concatenates cell line and drug latent vectors as input features
- **Dimensionality Reduction**: Reduces high-dimensional drug response data to manageable latent spaces

### 2. Neural Network Architecture
- **TensorFlow Implementation**: Deep learning model for IC50 prediction
- **Flexible Architecture**: Configurable hidden layers and activation functions
- **Regression Output**: Single output neuron for continuous IC50 values

### 3. Hyperparameter Optimization
- **Optuna Integration**: Automated hyperparameter search
- **Multi-objective**: Optimizes for both R² and NDCG metrics
- **Efficient Search**: Bayesian optimization for parameter space exploration

### 4. Comprehensive Evaluation
- **Spearman Correlation**: Per-sample and per-drug correlation analysis
- **NDCG (Normalized Discounted Cumulative Gain)**: Ranking-based evaluation
- **Cross-validation**: Robust model validation

### 5. Molecular Analysis
- **Chemical Similarity**: Tanimoto similarity using molecular fingerprints
- **Latent Space Analysis**: Cosine similarity between drug latent vectors
- **Correlation Studies**: Relationship between chemical and latent similarities

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd OncoRx
```

2. **Create conda environment**:
```bash
conda env create -f environment.yml
conda activate oncorx
```

3. **Install additional dependencies** (if needed):
```bash
pip install optuna tensorflow scikit-learn pandas numpy scipy matplotlib seaborn rdkit
```

## Usage

### 1. Hyperparameter Optimization

Run the Optuna optimization study:

```bash
python latent_model.py
```

This will:
- Load GDSC drug response data
- Perform matrix factorization using SVD
- Train neural networks with different hyperparameters
- Optimize for NDCG@10 metric
- Output best hyperparameters

### 2. Full Analysis Pipeline

Open and run the Jupyter notebook:

```bash
jupyter notebook my_model.ipynb
```

The notebook includes:
- Data loading and preprocessing
- Latent vector generation
- Model training with optimized hyperparameters
- Evaluation and visualization
- Molecular similarity analysis
- Gene expression mapping

### 3. Custom Analysis

Use the evaluation module for custom analyses:

```python
from cadrres_sc import evaluation

# Calculate Spearman correlation
per_sample_df, per_drug_df = evaluation.calculate_spearman(
    obs_df, pred_df, sample_list, drug_list
)

# Calculate NDCG
ndcg_df = evaluation.calculate_ndcg(obs_df, pred_df, k=10)
```

## Data Requirements

### Input Data Format
- **Drug Response Data**: Cell line × drug matrix with IC50 values
- **Gene Expression Data**: Gene × cell line expression matrix
- **Drug Information**: Chemical structures or PubChem IDs for similarity analysis


### Key Hyperparameters
- **SVD Components**: 5-20 (optimized)
- **Hidden Units**: 32-256 (step=32)
- **Learning Rate**: 1e-5 to 1e-2 (log scale)
- **Training Epochs**: 10 (configurable)

## Acknowledgments

- **GDSC**: Genomics of Drug Sensitivity in Cancer database
- **cadrres_sc**: normalization and evaluation metrics
- **IntOGen**: Cancer driver gene annotations
- **RDKit**: Chemical informatics toolkit
- **Optuna**: Hyperparameter optimization framework
- **TensorFlow**: Deep learning framework

---

**Note**: This project is for research purposes. Clinical applications require additional validation and regulatory approval.
