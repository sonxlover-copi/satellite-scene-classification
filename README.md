# Satellite Scene Classification

A computer vision workflow for a 20-class satellite image classification assignment. It compares compact CNN baselines with transfer learning from a pretrained ResNet-18 and exports predictions in the supplied submission format.

## What it covers

- Dataset and image integrity checks
- Reproducible train/validation split handling
- Compact CNN baselines and ResNet-18 fine-tuning
- Controlled learning-rate comparison using validation data
- Confusion matrix, error examples, and submission generation

## Data

The course archive and images are excluded because of their size. Obtain `week-5-competition-geo-satellites-classification.zip` from the course materials and extract it under `data/datasets/week5-satellites/` beside the notebook. Keep the original train, validation, and test CSVs and image directory layout intact.

## Run locally

Python 3.10 or newer is recommended. Install a PyTorch build compatible with your operating system and accelerator first if the default pip build is unsuitable, then install the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
jupyter lab
```

Open `satellite_classification.ipynb` and run all cells. The first run may download pretrained ResNet-18 weights. A GPU is recommended for full training.

## Validation status

All 13,997 referenced images were checked for presence and readability. The complete notebook path was smoke-tested on a reduced sample, including both learning-rate branches and submission generation. Full multi-epoch training on the complete dataset has not been run, so no full-dataset score is reported.

## Attribution

This is an adapted solution to an educational laboratory. Preserve the dataset and model licenses and the original assignment attribution included in the notebook.
