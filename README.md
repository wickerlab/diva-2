# DIVA-2 Project Repository

This repository contains the code, data preparation scripts, and Jupyter notebooks for the DIVA-2 machine learning project. The project encompasses both binary/anomaly detection and multiclass classification tasks, alongside meta-learning capabilities.

## 📁 Repository Structure

```text
diva-2/
├── .gitignore
├── clean_test/
│   ├── README.md
│   ├── new_metalearner.py
│   └── testv2.py
├── detector.ipynb
├── generate_data_folder.py
├── meta_db_universal.csv
├── multiclass.ipynb
├── poc_subsample.py
├── requirements.txt
├── scripts/
└── results/
    └── detector_checkpoint.pkl

```

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python installed on your system. To install the necessary project dependencies, use the provided requirements file:

```bash
pip install -r requirements.txt

```

---

## 🧠 Core Components

### Data Preparation

* **`meta_db_universal.csv`**: The primary metadata database file utilized across the project's models and scripts.
* **`generate_data_folder.py`**: A utility script designed to parse the metadata and structure the dataset directories for training and evaluation.
* **`poc_subsample.py`**: A Proof-of-Concept (PoC) script used for subsampling data, likely to create smaller, manageable datasets for rapid testing and prototyping.

**🔄 Regenerating the Metadata Database:**
If you need to regenerate or update the `meta_db_universal.csv` file, execute the following commands in order:

1. First, generate the data folder structure:
```bash
python generate_data_folder.py

```


2. Then, synchronize the metadata database using the provided script module:
```bash
python -m scripts.meta_db sync

```



### Modeling Notebooks

* **`detector.ipynb`**: A Jupyter Notebook dedicated to building, training, and evaluating the core detection model.
* **`multiclass.ipynb`**: A Jupyter Notebook focused on expanding the modeling to handle multiclass classification tasks.

---

## 💻 Computing Infrastructure

The development, training, and evaluation for this project were conducted on a machine with the following hardware and software specifications. When reproducing this work, similar or more powerful hardware is recommended (especially regarding GPU and VRAM) for optimal training times.

**Hardware Specifications:**

* **Processor (CPU):** 13th Gen Intel® Core™ i7-13620H (10 Cores, 16 Threads)
* **Memory (RAM):** 16 GB
* **Graphics (GPU):** NVIDIA GeForce RTX 3050 (6 GB VRAM)

**Software & Environment:**

* **Operating System:** Linux Mint 22.2 (Zara)
* **Python Version:** Python 3.10.20
* **CUDA Version:** 13.0 (NVIDIA Driver 580.159.03)
* **Virtual Environment:** Development was managed within a dedicated virtual environment.

---

## 📈 Results & Checkpoints

* **`results/detector_checkpoint.pkl`**: A serialized, pre-trained model checkpoint for the detector model. This can be loaded using Python's `pickle` library to run inferences without needing to retrain the model from scratch.
* **`results/multiclass_checkpoint.pkl`**: A serialized checkpoint for the multiclass model variant.