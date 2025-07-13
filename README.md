# Automated Cell Type Annotation Pipeline

This repository contains a fully automated pipeline for annotating cell types in single-cell RNA-sequencing (scRNA-seq) datasets from diseased human tissues. The solution was built from scratch to overcome the lack of labeled training data using a combination of unsupervised clustering, marker gene matching, and semi-supervised classification.

---

## Background & Problem

In real-world diseased scRNA-seq datasets, cell-type labels are often unavailable, making it difficult to train supervised models. Traditional manual annotation via marker gene expression is time-consuming and subjective.

Our task was to build an automated annotation pipeline **without any pre-labeled data**. This repository is the result of a creative and efficient workaround: we bootstrapped labels using marker genes and then trained a supervised classifier on only those confidently identified cells.

---

## Key Approach

1. **Preprocessing:**
   - Raw expression data is normalized and converted to `AnnData` format.
   - Dimensionality reduction using PCA and UMAP.
   - Clustering via Leiden algorithm.

2. **Hack: Marker-Guided Label Bootstrapping**
   - Known marker genes are used to find **high-confidence cells** within each cluster.
   - These cells serve as a **pseudo-labeled training set**.

3. **CellFM Classifier Training**
   - We use these confident cells to train a classifier using CellFM.
   - The trained model is then used to annotate the rest of the dataset.

---

## Datasets and Results

The pipeline was tested on 85,000+ cells from three diseased tissue datasets:

| Dataset       | Accuracy |
|---------------|----------|
| DCM_Heart     | 95.18%   |
| PSC_Liver     | 77.94%   |
| CKD_Kidney    | 91.71%   |

- **Total Processing Time:** Under 3 hours on a standard machine
- **Libraries Used:** Python, Scanpy, NumPy, Pandas (no high-level annotation libraries)

---

## Project Structure

Independent_One_Shot/
│
├── CellFM/ # CellFM model code and training logic
├── preprocessingData/ # Scanpy + Anndata preprocessing scripts
├── scType/ # Marker-based filtering and logic
├── output/ # Output CSVs and logs (excluded from Git)
├── figures/ # UMAP and clustering plots (excluded from Git)
├── checkpoint/ # Model checkpoints (excluded from Git)
├── .gitignore # Skips large or redundant files
├── *.ckpt, *.csv, *.out # Trained model files and results (excluded)
└── README.md # Project documentation


---

## Setup Instructions

### Requirements

- Python 3.8 or above
- Required Python packages:
  ```bash
  pip install scanpy anndata pandas numpy
## How It Works (In Short)

No labels → use marker genes to find reliable “seed” cells.

These cells → train classifier (CellFM).

Classifier → annotate entire dataset.

Saves days of manual work; accuracy on par with expert annotations.
