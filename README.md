# Breast Segmentation Project

This repository contains code for training and inference of a 3D U-Net model for segmentation on volumetric ultrasound (or video) data using PyTorch Lightning.

---

## 📁 Repository Structure

```
project_root/
│
├── src/                      # Source code
│   ├── dataset_seg.py        # Custom Dataset definitions for CSV-based loading
│   ├── data_datamodule_seg.py# PyTorch Lightning DataModule for train/val/test splits
│   ├── transforms_seg.py     # Data augmentation & preprocessing pipelines
│   ├── model_lightning_seg.py# LightningModule defining the 3D U-Net and training/predict logic
│   ├── train_lightning_kfold_seg.py # Script to train k-fold cross‑validation models
│   ├── inference_seg_folder.py # Script to run inference on a folder of MP4 videos (soft ensemble)
│   ├── inference_clas_folder.py # Script to run classification on a folder of MP4 videos (3 ensemble models)
│   ├── inference_pipeline.py # Script to run both segmentation and classification on a folder of MP4 videos (automatized workflow)
│   ├── loss.py               # Custom loss functions (e.g., Dice, BCE)
│   └── utils.py              # Helper functions and utilities
│
├── default_config_train_seg.yaml  # YAML configuration file for training and inference
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## 🚀 Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/your_username/breathing-segmentation.git
   cd breathing-segmentation
   ```
2. Create a Python environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## ⚙️ Configuration

All hyperparameters, paths, and training/inference settings are defined in `default_config_train_seg.yaml`. Edit this file to adjust:

* `model_opts`: architecture details
* `train_par`: training parameters (epochs, learning rate, batch size, early stopping, threshold)
* `dataset`: data directories, caching, augmentation flags

---

## 🎬 Training (k-Fold Cross Validation)

Use the training script to run k-fold training:

```bash
python src/train_lightning_kfold_seg.py \
  --config default_config_train_seg.yaml \
  --out_dir ./results_seg/experiment/ \
  --folds 10
```

To train the complete pipeline (segmentation and classification):

```bash
python inference_pipeline.py `
  --seg_config      "default_config_train_seg.yaml" `
  --seg_ckpt_dir    "Your-Path/Segmentacion_ckpts" `
  --video_dir       "Your-Path/Patient-ID" `
  --out_mask_dir    "Your-Path/Predictions/Patient-ID" `
  --seg_batch_size  1 `
  --cls_ckpt_paths  `
    "Your-Path/Classification_ckpts/densenet.ckpt" `
    "Your-Path/Classification_ckpts/mobilenet.ckpt" `
    "Your-Path/Classification_ckpts/vgg16.ckpt" `
  --n_samples       5 `
  --tol             0.2 `
  --video_ext       ".mp4" `
  --cls_batch_size  8 `
  --output_csv      "resultado_ensemble.csv"
```

This will:

1. Split your dataset into 10 folds (see CSV file paths in the YAML or script args).
2. Train one LightningModule per fold.
3. Save checkpoints as `kfold_1.ckpt`, ..., `kfold_10.ckpt` in the output directory.

---

## 🔍 Inference on Folder of MP4 Videos

After training, ensemble the k-fold models on new data:

```bash
python src/inference_seg_folder.py \
  --config default_config_train_seg.yaml \
  --ckpt_dir ./results_seg/experiment/ \
  --input_dir /path/to/your/mp4_folder \
  --out_dir ./results_seg/predictions \
  --batch_size 1
```

* **input\_dir** should contain your `.mp4` video files.
* Outputs will be saved as `*_mask.npy` in `out_dir`.
* You can load these NumPy masks or convert to NIfTI via `nibabel` if needed.

---

## 📂 Alternative CSV-Based Inference

If you prefer CSV-driven inference (with `dataset_seg.py` and `data_datamodule_seg.py`), see `inference_seg.py`:

```bash
python src/inference_seg.py \
  --config default_config_train_seg.yaml \
  --ckpt_dir ./results_seg/experiment/ \
  --test_csv ./data/test_list.csv \
  --out_dir ./predictions \
  --batch_size 1
```

---

## 📋 Dataset Organization

1. **CSV mode**: Two columns in your CSV: `wsi` (file path) and optionally `label`. The `DataModule` reads these to form train/val/test splits.
2. **Folder mode**: Place your `.mp4` files in one directory and use `inference_seg_folder.py`.

Ensure your data directory structure matches paths defined in the YAML config.

---

## 📝 Notes & Tips

* To adjust transforms, edit `get_transforms` in `transforms_seg.py`.
* To change model backbone or loss functions, modify `model_lightning_seg.py` and `loss.py`.
* Logging, checkpoints, and early stopping are managed by Lightning—consult the `train_lightning_kfold_seg.py` script for details.

---

## 🙋‍♂️ Contact

For questions or contributions, please open an issue or reach out to the maintainer.
