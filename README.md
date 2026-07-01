# YOLOCraft

Detection and segmentation of Minecraft mobs using YOLO and computer vision.

## Overview

YOLOCraft is a Deep Learning project focused on the automatic identification of Minecraft mobs in images. The goal is to train YOLO-family models to detect and/or segment in-game entities, enabling applications in computer vision, machine learning, and experimentation with game datasets.

Currently, the project focuses on training, evaluation, and testing of detection and segmentation models. In future versions, the trained model will be integrated into a web application for real-time inference.

The project was developed as an opportunity for hands-on study in:

* Computer Vision
* Object Detection
* Instance Segmentation
* YOLO model training
* Machine Learning pipeline organization

## Dataset

The datasets used are:

* Minecraft Mobs YOLO Dataset
* Source: Kaggle
* Classes corresponding to different Minecraft mobs

Dataset:

https://www.kaggle.com/datasets/dracotlw/minecraft-mobs-yolo-dataset/data

## Project Structure

```text
YOLOCraft/
│
├── data/
│   ├── minecraft_mobs/
│   └── minecraft_mobs-2/
│
├── notebooks/
│   ├── 1_exploration/
│   ├── 2_baseline/
│   └── 3_experiments/
│
├── src/
│   ├── config.py
│   ├── convert_dataset.py
│   ├── training_logger.py
│   ├── train_with_logging.py
│   ├── test_thresholds.py
│   ├── detector_gui.py
│   ├── dataset_manager.py
│   └── utils.py
│
├── scripts/
│   └── download_dataset.py
│
├── pretrained_models/
├── models/
├── training_logs/
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone https://github.com/your-user/YOLOCraft.git
cd YOLOCraft
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment:

### Linux / macOS

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Kaggle Setup

The project uses the official Kaggle CLI to download the dataset automatically.

After installing the dependencies, run:

```bash
kaggle auth login
```

A link will be shown in the terminal.

1. Open the link in your browser;
2. Sign in to your Kaggle account;
3. Authorize access;
4. Return to the terminal.

The credentials are stored automatically and you will not need to repeat this process.

To verify that authentication was successful:

```bash
kaggle datasets list -s minecraft
```

## Dataset Download

After authenticating your account:

```bash
python scripts/download_dataset.py
```

The script will:

* Check whether the dataset already exists locally;
* Download only when needed;
* Organize the project files automatically.

The data is stored in:

```text
data/
```

## Training

Training is done in the experiment notebooks under `notebooks/3_experiments/`, which record each run via `src/training_logger.py`. A scripted entry point is also available:

```bash
python -m src.train_with_logging
```

Each training run produces:

* Model weights
* Validation metrics
* Learning curves
* A logged entry in `training_logs/`

Run outputs are saved under:

```text
runs/
```

## Dataset Manager and Detector

Two PyQt applications support the workflow:

```bash
python -m src.dataset_manager
python -m src.detector_gui
```

* `dataset_manager` browses classes and samples, and exports a curated YOLO dataset.
* `detector_gui` loads a model and runs detection on an uploaded image.

Clicking an image in the dataset manager sends it to an open detector window.

## Goals

### Current Phase — Object Detection

* Automatically detect mobs in images using YOLO-family models;
* Evaluate the performance of modern object detection architectures;
* Compare different YOLO models (Nano, Small, Medium, etc.);
* Evaluate metrics such as Precision, Recall, mAP50, and mAP50-95;
* Build a reproducible pipeline for training, validation, and inference;
* Investigate the impact of different hyperparameters on model performance.

### Next Phase — Segmentation with OpenCV

* Use the bounding boxes produced by YOLO as regions of interest (ROI);
* Apply classical segmentation techniques with OpenCV;
* Compare different segmentation algorithms;
* Evaluate the quality of the generated masks;
* Investigate hybrid approaches combining Deep Learning and Digital Image Processing.

### Future Phase — Web Application

* Develop a web application for inference;
* Allow users to upload images;
* Display detections in real time;
* Display the segmented masks produced by OpenCV;
* Make the trained model available online;
* Build a friendly interface for demonstrating results.

## Technologies Used

* Python
* PyTorch
* Ultralytics YOLO
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Jupyter Notebook
* Kaggle CLI
* Git and GitHub

## Roadmap

### Version 1.0 — Detection with YOLO

* [x] Initial project structure
* [x] Automated dataset download
* [x] Exploratory data analysis
* [x] Class balance check
* [x] Visual validation of annotations
* [ ] Baseline training (YOLO26n)
* [ ] Performance evaluation
* [ ] Inference tests
* [ ] Comparison between YOLO architectures
* [ ] Selection of the final detection model

### Version 2.0 — Segmentation with OpenCV

* [ ] Region of interest (ROI) extraction
* [ ] Threshold segmentation
* [ ] Otsu segmentation
* [ ] Canny segmentation
* [ ] GrabCut segmentation
* [ ] Watershed segmentation
* [ ] Comparison between segmentation methods
* [ ] Qualitative evaluation of results
* [ ] YOLO + OpenCV integration

### Version 3.0 — Web Application

* [ ] Inference API development
* [ ] Web application
* [ ] Image upload
* [ ] Bounding box visualization
* [ ] Segmented mask visualization
* [ ] Results dashboard
* [ ] Cloud deployment

## Expected Results

* High accuracy in mob identification;
* Automated training and inference pipeline;
* A foundation for future projects involving computer vision in game environments;
* Future integration with web applications.

## License

This project is intended for educational purposes, research, and learning in computer vision and Deep Learning.
