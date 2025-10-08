# TonotopyCNN — Neuroscience-Inspired Audio Classification

This project explores **neuroscience-inspired convolutional networks** for sound classification using the **ESC-50 dataset**.
It compares a standard baseline CNN with a **tonotopic / Gabor-initialized CNN** that mimics the **frequency-selective organization of the auditory cortex**.

## Motivation

In the human auditory system, neurons are spatially organized by the frequencies they respond to — a phenomenon known as **tonotopy**.
This project aims to encode a similar structure into early convolutional layers, using **Gabor-like filters** and **frequency-based regularization**, to investigate whether such biologically inspired priors improve performance and interpretability.

## Results

### Performance Comparison
[[![plot](graphs /comparison_curves.png)](https://graphs/baseline_conv1_ep60.png)](https://graphs/comparison_curves.png)

### Baseline CNN Visualizations

Conv1 Filters at Epoch 60
[![graphs /baseline_conv1_ep60.png]](https://graphs/baseline_conv1_ep60.png)

Tonotopic Organization Analysis
[![graphs /baseline_centers_1759903798.png]](https://graphs/baseline_centers_1759903798.png)
[![graphs /baseline_preferred_freqs_1759903798.png]](https://graphs/baseline_preferred_freqs_1759903798.png)
[![graphs /baseline_tuning_heat_1759903798.png]](https://graphs/baseline_tuning_heat_1759903798.png)

### Neuro-Inspired CNN Visualizations

Gabor-Initialized Conv1 Filters at Epoch 60
![graphs /neuro_conv1_ep60.png]

Tonotopic Organization Analysis
![graphs /neuro_centers_1759903797.png]
![graphs /neuro_preferred_freqs_1759903798.png]
![graphs /neuro_tuning_heat_1759903797.png]
