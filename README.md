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

## Credits

This project uses datasets and resources provided by the [MIDOG 2025 Challenge](https://midog2025.deepmicroscopy.org/), which builds on the success of previous MIDOG challenges to advance AI-assisted cancer diagnosis.

Special thanks to the MIDOG organizers for their efforts in curating high-quality datasets and promoting research in histopathological image analysis.