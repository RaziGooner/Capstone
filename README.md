# Active Learning-Based Image Segmentation for Fault Detection

**Author:** Mohammad Raziuddin Chowdhury  
**Course:** Capstone Project  
**Date:** 4th April, 2025  

## Overview

This project implements an Active Learning framework for semantic segmentation of surface defects using the KolektorSDD2 dataset. We compare baseline training on the full dataset vs. an active learning loop that iteratively selects the most informative samples to label. An initial model for the active learning is trained on 50% of the total training data which will be used to do the active learning iterations.

The core model is a U-Net architecture with an EfficientNet-B4 as encoder, trained with Dice + Focal loss.

```
Capstone/
├── src/
│   ├── model.py                 # U-Net model with EfficientNet-B4 (for baseline and active learning initial model)
│   ├── utils.py                 # Training, plotting, visualization etc.functiions 
│   ├── prepare_data.py          # Dataset preparation utilities
│   ├── aug_torchdataset.py      # Dataset class + data augmentation pipelines
│   ├── evaluation.py            # Metric evaluation functions (IoU, Dice, etc.)
│   └── active_learning.py       # Active learning loop logic
│
├── models/
│   ├── al_model.pt
│   ├── baseline_model.pt
│   ├── al_model_initial.pt
│
├── data/
│   ├── KolektorSDD2.zip         # (Added instructions for download)
│   ├── train_images/            # Training images
│   ├── train_masks/             # Training masks
│   ├── test_images/             # Test images
│   ├── test_masks/              # Test masks
│   ├── train_data.csv           # Metadata for training
│   ├── test_data.csv            # Metadata for test
│   ├── al_model.pt
│   ├── baseline_model.pt
│   ├── al_model_initial.pt
│
├── figure/
│   ├── baseline_model_loss.jpg
│   ├── initial_al_model_loss.jpg
│
├── main.IPYNB                   # Runs baseline training and active learning
├── model_trainer.py             # prepare data and train and log
├── test_models.py               # test models on test data and show predictions
├── requirements.txt
└── README.md
```



## Installation Instructions
1. Clone the Repository
   ```
   git clone https://github.com/RaziGooner/Capstone.git
   cd Capstone
   ```
2. Set up the Environment
   ```
   pip install -r requirements.txt
   ```


## Dataset Preparation & Training
Download the KolektorSDD2 dataset from https://www.vicos.si/resources/kolektorsdd2/  

The steps to prepare data is included in the run_trainer.py.  
This file will:  
1. prepare data for training,
2. Create the necessary data splits and dataloaders,
3. Initialize the baseline model, initial model for active learning and train them with the default hyperparameters,
4. Will set the active learning loop off following the default hyperparameters,
5. Save both the baseline and active learnig model (after active learning iterations)
6. Evaluate both the models on the test set defined by the dataset authors
7. will save all the necessary and important information in the log file.


Need to mention a base directory and the path to the downloaded zip file(dataset) to run the run_trainer.py file

```
python run_training.py --base_dir Capstone --zip_path ./Capstone/KolektorSDD2.zip
```


## Testing

The test_moddels file will give you the predictions for 3 samples from the test set for both the baseline model and the active learning model


## Notebook 'main.IPYNB'
This main notebook has all the outputs for a run and will provide a good overlook.

## Results

## Model Evaluation Metrics

| Metric    | Baseline Model | Active Learning Model |
|-----------|----------------|------------------------|
| IoU       | 0.6634         | 0.6634                 |
| Dice      | 0.7576         | 0.7576                 |
| Accuracy  | 0.9989         | 0.9989                 |
| Precision | 0.8593         | 0.8693                 |
| Recall    | 0.7624         | 0.7765                 |

