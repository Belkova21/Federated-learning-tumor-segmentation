# Federated Learning Tumor Segmentation

This project implements a federated learning pipeline for **brain tumor segmentation** from T1-weighted MRI images using the **U-Net** architecture. The aim is to demonstrate that competitive segmentation performance can be achieved **without sharing patient data**, making the approach suitable for multi-institutional medical collaboration.

## 🔍 Project Overview

- **Model:** Custom 5-level `DynamicUNet` for binary segmentation
- **Dataset:** Preprocessed `.mat` MRI slices (3,064 images from 233 patients)
- **Clients:** Simulated federated setup with 2 virtual clients (representing hospitals)
- **Framework:** Built with [Flower](https://flower.dev) for federated learning orchestration
- **Training Modes:**
  - **Centralized Learning (CL):** All data is used in one model
  - **Federated Learning (FL):** Each client trains locally, with **top-k sparsification** for communication efficiency

---

## Run Locally

### 1. Install requirements

```bash
pip install -r requirements.txt
```

### 2. Download and prepare the dataset

```bash
Open and run dataset_source.ipynb
```

This script downloads and unpacks the dataset.

---

## Training

### ▶Centralized Learning

```bash
python main.py
```

### Federated Learning

**Start the server:**

```bash
python server.py
```

**Start clients (in separate terminals):**

```bash
python client.py 0
python client.py 1
```
