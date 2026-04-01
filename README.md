# Deep Learning for Atypical Mitotic Figure Classification

## Overview

This project focuses on binary classification of mitotic figure patches into:

- AMF: Atypical Mitotic Figures
- NMF: Normal Mitotic Figures

Detecting AMF is clinically relevant but difficult because atypical samples are rare and visually subtle in H&E-stained histopathology images. The pipeline in this repository addresses that challenge through curated data preparation, class-balancing augmentation, baseline model benchmarking, and attention-based model improvement.

---

## Key Highlights

- Dataset size: 14,000+ RGB image patches (224 x 224)
- Balanced training strategy using targeted 4x AMF augmentation
- Baseline benchmark across multiple architectures
- Final backbone choice: DenseNet-121
- Best convolutional attention in experiments: SE (Squeeze-and-Excitation)

---

## Repository Structure

| Folder | Purpose |
|---|---|
| `Dataset/` | Original source datasets and CSV metadata |
| `Augmentation/` | Data augmentation and train/val/test split notebooks |
| `AugmentedDataset/` | Final processed training/validation/testing image sets |
| `BaselineModel/` | Baseline architecture training notebooks and result files |
| `Attentions/` | Attention module experiments (SE, CBAM, CCA, ECA, PSA, etc.) |
| `Position/SE1D/` | Non-local attention position search experiments |
| `Report&PPT/` | Report and presentation artifacts |

---

## Dataset

### Data Sources

The dataset combines samples from:

- MIDOG25
- AMI-BR
- MIDOG21 (via AMI-BR)
- TUPAC16 (via AMI-BR)

All samples are image patches centered on mitotic figures.

### Split Summary

| Split | Approx. Samples | Notes |
|---|---:|---|
| Training | ~9,000 | Includes augmented AMF samples |
| Validation | ~3,000 | Original images only |
| Test | ~3,000 | Original images only |

Validation and test sets intentionally use non-augmented data to preserve unbiased evaluation.

### Links

- Original curated dataset: [Zenodo](https://zenodo.org/records/15188326)
- Augmented training dataset: [Kaggle](https://www.kaggle.com/datasets/lostluinor/mitoticfigure-spiltandaugmenteddataset)

---

## Class Imbalance Handling

AMF samples are naturally underrepresented. To improve class balance in training:

- Augmentation is applied only to AMF training samples
- Each AMF image generates 4 additional variants
- Effective outcome is approximately +6,000 AMF samples
- Final training balance is approximately 1:1 (AMF:NMF)

Augmentation operations include controlled rotation, flips, brightness/contrast variation, and sharpness adjustment.

---

## Modeling Workflow

1. Benchmark multiple baseline architectures.
2. Select the strongest backbone using accuracy, precision, recall, F1-score, and loss.
3. Add and compare multiple convolutional attention modules.
4. Pick the best attention design and tune placement using SE1D position experiments.

---

## Baseline Model Comparison

DenseNet-121 was selected as the main backbone due to strong and stable overall performance.

| Model | Accuracy (%) | Loss | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|
| EfficientNet-B0 | 85.29 | 0.1035 | 0.8500 | 0.8500 | 0.8500 |
| **DenseNet-121** | **89.02** | **0.0900** | **0.8904** | **0.8902** | **0.8903** |
| ViT-B16 | 87.16 | 0.0960 | 0.8720 | 0.8715 | 0.8718 |
| ConvNeXt-Tiny | 89.31 | 0.0866 | 0.8890 | 0.8931 | 0.8906 |
| MobileNetV2-100 | 87.19 | 0.0976 | 0.8543 | 0.8690 | 0.8575 |
| NASNet-Large | 82.42 | 0.1471 | 0.7308 | 0.8242 | 0.7679 |
| RegNetY-16 | 87.16 | 0.0960 | 0.8720 | 0.8715 | 0.8718 |
| XceptionNet | 89.31 | 0.0866 | 0.8890 | 0.8931 | 0.8906 |

---

## Training Configuration

| Parameter | Value |
|---|---|
| Input size | 224 x 224 |
| Batch size | 32 |
| Epochs | 50 |
| Optimizer | Adam |
| Loss | Focal Loss ($\alpha=0.75$, $\gamma=2.0$) |
| Final activation | Softmax |
| Backbone | DenseNet-121 (ImageNet pretrained) |
| Attention | SE + SE1D |

Primary evaluation focus is precision, since reducing false positives is important for atypical mitosis screening.

---

## Attention Module Experiments

SE achieved the best overall metrics among tested attention variants and combinations.

| Model | Accuracy | Loss | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|
| CBAM | 0.8859 | 0.0874 | 0.8855 | 0.8859 | 0.8857 |
| CCA | 0.8888 | 0.0881 | 0.8833 | 0.8888 | 0.8854 |
| ECA | 0.8841 | 0.0986 | 0.8849 | 0.8841 | 0.8845 |
| GAM | 0.8741 | 0.0912 | 0.8810 | 0.8741 | 0.8771 |
| PSA | 0.8910 | 0.0885 | 0.8817 | 0.8877 | 0.8840 |
| **SE** | **0.8942** | **0.0874** | **0.8937** | **0.8942** | **0.8939** |
| SE+CCA | 0.8856 | 0.0941 | 0.8856 | 0.8856 | 0.8866 |
| SE+PSA | 0.8841 | 0.0982 | 0.8901 | 0.8841 | 0.8867 |
| Self-Att | 0.8795 | 0.0875 | 0.8782 | 0.8795 | 0.8788 |
| Triplet | 0.8802 | 0.0908 | 0.8849 | 0.8802 | 0.8823 |

---

## Report and Presentation Files

The `Report&PPT` folder currently includes:

- `DL_MitoticClassification_IEEEReport.pdf`
- `DL_MitoticFigureClassification_LatexReport.tex`
- `DL_MitoticFigureClassification_ppt.pdf`
- `DL_MitoticFigureClassification_ppt.pptx`

---

## Credits

This work uses resources from the [MIDOG 2025 Challenge](https://midog2025.deepmicroscopy.org/) and related datasets.

Thanks to the MIDOG organizers and contributors for supporting reproducible research in computational pathology.