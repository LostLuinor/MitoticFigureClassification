# Deep Learning Model for Identification of Atypical Mitotic Figures

## Project Overview

Atypical Mitotic Figures (AMF) are rare but highly relevant markers of tumour aggressiveness in histopathology. Distinguishing AMF from Normal Mitotic Figures (NMF) is challenging due to their subtle morphological differences and low frequency in whole-slide images. Reliable automated detection requires high-quality, well-annotated datasets that capture this variability. This project uses a curated and augmented dataset of mitotic figure image patches to support robust binary classification between AMF and NMF.

---

## Dataset Description

### Overview

This project uses the **MIDOG Mitotic Figures Binary Classification Dataset – Balanced & Augmented**, a curated collection designed specifically for distinguishing Normal Mitotic Figures (NMF) and Atypical Mitotic Figures (AMF). The dataset integrates samples from several established histopathology challenges and repositories, including:

- **MIDOG25**
- **AMI-BR** dataset, which itself includes:
	- MIDOG21
	- TUPAC16

The dataset contains over **14,000 high-resolution PNG images (224×224)**, accompanied by metadata and documentation. All samples represent mitotic figure patches extracted from H&E-stained breast cancer whole-slide images.

---

### Dataset Structure

A stratified split ensures equal representation of both classes across the three subsets:

| Split      | Approx. Samples | Notes                                 |
|------------|-----------------|---------------------------------------|
| Training   | ~9,000          | Contains augmented AMF samples         |
| Validation | ~3,000          | Only original images (no augmentation) |
| Test       | ~3,000          | Only original images (no augmentation) |

Validation and test sets contain exclusively non-augmented images to preserve unbiased evaluation conditions.

**Dataset access:**  
[Original curated dataset](https://zenodo.org/records/15188326)

---

## Class Imbalance Handling & Augmented Dataset

Atypical mitoses are naturally underrepresented in real histopathology data. To counter this imbalance, a **4× augmentation strategy** was applied only to AMF training images. This augmentation does not affect the validation or test sets.

Each original AMF image generates four additional variants, created using PIL-based augmentation with a fixed random seed for reproducibility. Transformations applied include:

- **Random Rotation**
	- 15°–30°
	- White background fill
- **Horizontal Flip + Brightness Adjustment**
	- Horizontal flip
	- Brightness scaled between 0.9×–1.1×
- **Vertical Flip + Contrast Adjustment**
	- Vertical flip
	- Contrast scaled between 0.9×–1.1×
- **Rotation + Sharpness Enhancement**
	- 5°–15° rotation
	- Sharpness adjustment between 0.9×–1.1×

This results in approximately **6,000 additional AMF samples**, yielding a balanced **1:1 ratio of AMF to NMF** in the training dataset.

**Augmented dataset:**  
[Augmented training dataset](https://www.kaggle.com/datasets/lostluinor/mitoticfigure-spiltandaugmenteddataset)

---


## Image Format & Resolution

- **Format:** PNG
- **Resolution:** 224 × 224 pixels
- **Color Space:** RGB
- **Content:** Isolated patches centered on mitotic figures

---

## Baseline Model Selection & Comparison

To identify the best baseline architecture, several CNN models—including DenseNet, MobileNet, ConvNeXt, EfficientNet, ViT, NASNet, RegNet, and XceptionNet—were evaluated on the dataset. Performance metrics such as accuracy, loss, precision, recall, and F1-score were compared across models.

Based on the results, **DenseNet-121** was selected as the backbone for further development due to its strong overall performance, achieving high accuracy and balanced precision, recall, and F1-score.

**Model Performance Comparison:**

| Model             | Accuracy (%) | Loss    | Precision | Recall   | F1-score |
|-------------------|-------------|---------|-----------|----------|----------|
| EfficientNet-B0   | 85.29       | 0.1035  | 0.8500    | 0.8500   | 0.8500   |
| **DenseNet-121**  | **89.02**   | **0.0900** | **0.8904** | **0.8902** | **0.8903** |
| ViT-B16           | 87.16       | 0.0960  | 0.8720    | 0.8715   | 0.8718   |
| ConvNeXt-Tiny     | 89.31       | 0.0866  | 0.8890    | 0.8931   | 0.8906   |
| MobileNetv2-100   | 87.19       | 0.0976  | 0.8543    | 0.8690   | 0.8575   |
| NASNet-Large      | 82.42       | 0.1471  | 0.7308    | 0.8242   | 0.7679   |
| RegNetY-16        | 87.16       | 0.0960  | 0.8720    | 0.8715   | 0.8718   |
| XceptionNet       | 89.31       | 0.0866  | 0.8890    | 0.8931   | 0.8906   |

---

## Model Training & Evaluation Summary

The model was trained using images resized to **224 × 224** pixels to match the DenseNet-121 input format. Training was performed for **50 epochs** with a **batch size of 32** using the **Adam optimizer**. To address class imbalance, **Focal Loss** ($\alpha = 0.75$, $\gamma = 2.0$) was employed. The final classification layer used **Softmax activation**.

**Channel attention** was introduced via a Squeeze-and-Excitation (SE) module, and a **non-local attention block** (SE1D) was added at the experimentally determined optimal position. The backbone architecture was **DenseNet-121 pretrained on ImageNet**.

**Key hyperparameters and configurations:**

| Parameter               | Description                        | Value                                 |
|------------------------|------------------------------------|---------------------------------------|
| Input Image Size       | Resized image resolution           | 224 × 224                             |
| Batch Size             | Samples per batch during training  | 32                                    |
| Epochs                 | Total training iterations          | 50                                    |
| Optimizer              | Optimization algorithm             | Adam                                  |
| Loss Function          | Training loss function             | Focal Loss ($\alpha=0.75$, $\gamma=2.0$) |
| Activation Function    | Final layer activation             | Softmax                               |
| Backbone Architecture  | Pre-trained CNN backbone           | DenseNet-121 (ImageNet)               |
| Attention Mechanism    | Channel attention module           | SE (Squeeze-and-Excitation)           |
| Non-Local Attention    | Feature-wise self-attention        | SE1D                                  |

The model was evaluated using four metrics: **accuracy, precision, recall, and F1 Score**. **Precision** was the primary metric, as it is crucial for identifying false positives when differentiating atypical mitotic figures from other similar cells. The other metrics provided additional insights into the overall classification performance.

---

## Credits

This project uses datasets and resources provided by the [MIDOG 2025 Challenge](https://midog2025.deepmicroscopy.org/), which builds on the success of previous MIDOG challenges to advance AI-assisted cancer diagnosis.

Special thanks to the MIDOG organizers for their efforts in curating high-quality datasets and promoting research in histopathological image analysis.