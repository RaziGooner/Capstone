# Active Learning-Based Image Segmentation for Fault Detection

**Author:** Mohammad Raziuddin Chowdhury  
**Course:** Capstone Project  
**Date:** 4th April, 2025  

## Overview

This project implements an Active Learning framework for semantic segmentation of surface defects using the KolektorSDD2 dataset. We compare baseline training on the full dataset vs. an active learning loop that iteratively selects the most informative samples to label. An initial model for the active learning is trained on 50% of the total training data which will be used to do the active learning iterations.

The core model is a U-Net architecture with an EfficientNet-B4 as encoder, trained with Dice + Focal loss.


'''<pre> Capstone/ ├── src/ │ ├── model.py # U-Net++ model with EfficientNet-B4 │  ├── utils.py # Training loop, plotting, and visualization functions │  ├── prepare_data.py # Dataset preparation utilities │ ├── aug_torchdataset.py # Dataset class + data augmentation pipelines │ ├── evaluation.py # Metric evaluation functions (IoU, Dice, etc.) │ └── active_learning.py # Active learning loop logic │ ├── Capstone/ │ ├── KolektorSDD2.zip # (Add instructions for download) │ ├── train_images/ # Training images │ ├── train_masks/ # Training masks │ ├── test_images/ # Test images │ ├── test_masks/ # Test masks │ ├── train_data.csv # Metadata for training │ ├── test_data.csv # Metadata for test │ ├── model/ │ │ ├── baseline_model.pt │ │ ├── al_model_initial.pt │ │ └── al_model.pt │ └── figure/ │ └── baseline_model_loss.jpg │ ├── main.py # Runs baseline training and active learning ├── requirements.txt └── README.md </pre>
