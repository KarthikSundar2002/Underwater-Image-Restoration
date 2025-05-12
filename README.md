# Underwater Image Restoration

This repository contains code for training and evaluating deep learning models for underwater image restoration. The project supports multiple model architectures, custom loss functions, and experiment tracking with Weights & Biases (wandb).

---

## Table of Contents

- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Dataset Preparation](#dataset-preparation)
- [Available Models](#available-models)
- [Training & Evaluation](#training--evaluation)
- [Experiment Tracking](#experiment-tracking)
- [Utilities](#utilities)
- [Notes](#notes)

---

## Project Structure

```
underwater-image-restoration/
│
├── main.py                # Main entry point for training/evaluation
├── args.py                # Command-line argument definitions
├── requirements.txt       # Python dependencies
├── src/                   # All source code
│   ├── Models/            # Model architectures (SpectralTransformer, AST, etc.)
│   ├── model/             # Model building blocks
│   ├── DataManipulation/  # Data loading, augmentation, and management
│   ├── Losses/            # Custom loss functions
│   ├── utils/             # Logging, visualization, wandb integration
│   └── Secrets/           # (For Kaggle API key, not versioned)
├── logs/                  # Training logs and checkpoints
├── wandb/                 # wandb experiment logs
├── data/                  # Place your datasets here
├── README.md
└── ...
```

---

## Setup & Installation

1. **Python Environment**

   - Use Python 3.10.16 (recommended).
   - Create a virtual environment:
     ```bash
     python3.10 -m venv .venv
     source .venv/bin/activate
     ```

2. **Install Dependencies**

   - Install all required packages:
     ```bash
     pip install -r requirements.txt
     ```

   - For CUDA-enabled PyTorch (if using GPU):
     ```bash
     pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
     ```

3. **Additional Tools**
   - [wandb](https://wandb.ai/) for experiment tracking:
     ```bash
     pip install wandb
     wandb login
     ```
   - KaggleHub for dataset downloads:
     - Place your `kaggle.json` API key in `src/Secrets/Secret/kaggle.json`.

---

## Dataset Preparation

- **Manual Placement:**  
  Place your datasets in the `data/` directory.  
  The expected structure for training and testing is:
  ```
  data/
    uw_data/
      uw_data/
        train/
          a/   # Raw images
          b/   # Reference images
        test/
          a/   # Raw test images
          b/   # Reference test images
  ```

- **Automatic Download (Kaggle):**  
  The code supports downloading datasets from Kaggle using the `DownloaderKaggle` utility.  
  - Ensure your Kaggle API key is at `src/Secrets/Secret/kaggle.json`.
  - The downloader can fetch datasets like `larjeck/uieb-dataset-raw` and `larjeck/uieb-dataset-reference`.

---

## Available Models

The following model architectures are available (see `src/Models/__init__.py`):

- `SpectralTransformer`
- `AST`
- `NewModel`
- `NewBigModel`
- `NewBigFRFNModel`

You can specify the model with the `-a` or `--arch` argument.

---

## Training & Evaluation

### **Training Example**

```bash
python main.py -a SpectralTransformer --lr 0.0003 --max-epoch 2500 --lossf L1withColor
```

- All training/evaluation arguments are defined in `args.py`.
- Key arguments:
  - `-a`, `--arch`: Model architecture (default: SpectralTransformer)
  - `--lr`: Learning rate
  - `--max-epoch`: Number of epochs
  - `--lossf`: Loss function (e.g., L1, L1withColor, mix, fflmix, charbonnier, perceptual)
  - `--train-batch-size`, `--test-batch-size`: Batch sizes
  - `--optim`: Optimizer (e.g., adam, adamw)
  - `--use-wandb`: Enable/disable wandb logging

### **Evaluation Example**

```bash
python main.py --evaluate --load-weights path/to/checkpoint.pth
```

---

## Experiment Tracking

- **Weights & Biases (wandb):**
  - Enabled by default. Disable with `--use-wandb False`.
  - Logs metrics, losses, and model checkpoints.

---

## Utilities

- **Data Augmentation:**  
  `src/DataManipulation/DataAugmentor.py` provides configurable augmentation.
- **Custom Losses:**  
  See `src/Losses/` for implemented loss functions.
- **Visualization:**  
  `src/utils/Visualiser.py` for model output visualization.
- **Logging:**  
  Training logs are saved in the `logs/` directory.

---

## Notes

- All code files are in the `src/` directory.
- Place your datasets in the `data/` directory or use the Kaggle downloader.
- For Kaggle downloads, ensure your API key is present at `src/Secrets/Secret/kaggle.json`.
- For more training/evaluation options, see `args.py`.

---

**For any issues or questions, please open an issue or contact the maintainers.**
