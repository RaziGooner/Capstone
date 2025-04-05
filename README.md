# Active Learning-Based Image Segmentation for Fault Detection

**Author:** Mohammad Raziuddin Chowdhury  
**Course:** Capstone Project  
**Date:** 4th April, 2025  

## 🧠 Overview

This project implements an Active Learning framework for semantic segmentation of surface defects using the KolektorSDD2 dataset. We compare baseline training on the full dataset vs. an active learning loop that iteratively selects the most informative samples to label. An initial model for the active learning is trained on 50% of the total training data which will be used to do the active learning iterations.

The core model is a U-Net architecture with an EfficientNet-B4 as encoder, trained with Dice + Focal loss.
