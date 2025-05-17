# ECSC: Energy-Calibrated Semantic Communication

**Authors:** Ziyu Wang*, Yiming Du*, Rui Ning*, Daniel Takabi†, and Lusi Li*
<br>
*Department of Computer Science, Old Dominion University, Norfolk, VA, 23529, USA*
<br>
*†School of Cybersecurity, Old Dominion University, Norfolk, VA, 23529, USA*

**Paper Link:** (TBD)
**Code Repository:** [https://github.com/sunway677/ECSC]

## Overview

Ensuring robust semantic preservation under noisy channel conditions remains a fundamental challenge in semantic communication. Most existing approaches, which rely on either task-specific supervision or pixel-level reconstruction loss, often fail to maintain semantic consistency when faced with both channel noise and inherent semantic distortions.

This repository contains the official PyTorch implementation for the paper "ECSC: Energy-Calibrated Semantic Communication". ECSC is an end-to-end framework that integrates a multi-scale U-Net autoencoder for hierarchical feature learning, lightweight compression-decompression modules for efficient transmission, and a trainable energy-based model (EBM) for semantic calibration. Our framework uniquely enforces hierarchical alignment through semantic-aware reconstruction. An energy-weighted reconstruction loss in terms of images and a multi-scale feature reconstruction loss enforces hierarchical alignment in terms of features. A sparsity regularization is applied to feature levels. Our EBM, trained via margin-based contrastive learning, enhances semantic alignment between input and reconstructed images by leveraging a pre-trained CLIP vision encoder as a semantic estimator. Furthermore, we propose a K-nearest neighbor topology regularizer to preserve latent feature relationships without requiring labeled data, ensuring structural consistency in the semantic space.

Experiments on CIFAR-10 over AWGN channels demonstrate ECSC’s superiority, especially in low-SNR scenarios, delivering improved semantic fidelity, reconstruction quality, and robustness to both distortions.

## Key Features

* **Energy-Calibrated Semantic Communication (ECSC):** A novel framework for robust semantic communication.
* **Multi-Scale U-Net Autoencoder:** Employs a U-Net architecture with ResNet backbones for hierarchical feature learning.
* **Lightweight Compression/Decompression:** Efficient modules for feature compression before transmission and decompression at the receiver.
* **Trainable Energy-Based Model (EBM):** Calibrates semantic representations by assigning low energy to semantically aligned pairs and high energy to mismatched ones.
* **CLIP-Guided Semantic Estimation:** Uses a pre-trained CLIP vision encoder to provide semantic guidance to the EBM.
* **Hierarchical Alignment:**
    * Energy-weighted reconstruction loss for images.
    * Multi-scale feature reconstruction loss for feature consistency.
* **Sparsity Regularization:** Applied to feature levels to promote efficient representations.
* **K-Nearest Neighbor (KNN) Topology Regularizer:** Preserves latent feature relationships in the semantic space without labeled data.
* **Robust Performance:** Demonstrates superior performance on CIFAR-10, especially in low Signal-to-Noise Ratio (SNR) conditions over AWGN channels.

## System Architecture

[ECSC Architecture](./ECSC.pdf)

## Requirements

* Python 3.x
* PyTorch
* Transformers (for CLIP)
* NumPy
* Matplotlib (for visualization)
* Scikit-image (for metrics like SSIM, PSNR)
* tqdm (for progress bars)
