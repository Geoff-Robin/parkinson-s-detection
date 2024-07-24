# Parkinson's Detection

A machine learning project for detecting Parkinson's Disease using voice and speech analysis. This repository contains the necessary code, data, and instructions for training and evaluating a model for Parkinson's Disease detection.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Model](#model)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

Parkinson's Disease is a neurodegenerative disorder that affects movement and speech. This project aims to detect Parkinson's Disease by analyzing voice and speech features using machine learning techniques.

## Features

The model uses the following 22 voice features to detect Parkinson's Disease:

1. **MDVP:Fo(Hz)**: Mean fundamental frequency of the voice. It represents the average pitch of the voice signal.

2. **MDVP:Fhi(Hz)**: Maximum fundamental frequency of the voice. It indicates the highest pitch observed in the voice signal.

3. **MDVP:Flo(Hz)**: Minimum fundamental frequency of the voice. It represents the lowest pitch in the voice signal.

4. **MDVP:Jitter(%)**: Percent of jitter, which measures the frequency variation of the voice signal as a percentage.

5. **MDVP:Jitter(Abs)**: Absolute jitter, a measure of the variability in the frequency of the voice signal.

6. **MDVP:RAP**: Relative Average Perturbation, a measure of short-term frequency perturbation in the voice signal.

7. **MDVP:PPQ**: Pitch Perturbation Quotient, a measure of the irregularity in pitch over short periods.

8. **Jitter:DDP**: Difference of Differences of Pitch Periods, a measure of pitch period variability.

9. **MDVP:Shimmer**: The average amplitude variation in the voice signal, which reflects the voice’s loudness stability.

10. **MDVP:Shimmer(dB)**: Shimmer in decibels, which represents the amplitude variation in a logarithmic scale.

11. **Shimmer:APQ3**: Amplitude Perturbation Quotient over three periods, a measure of short-term amplitude perturbation.

12. **Shimmer:APQ5**: Amplitude Perturbation Quotient over five periods, another measure of amplitude perturbation over a longer duration.

13. **MDVP:APQ**: Amplitude Perturbation Quotient, a general measure of amplitude variability in the voice signal.

14. **Shimmer:DDA**: Double Difference of Amplitude, a measure of amplitude perturbation in the voice signal.

15. **NHR**: Noise-to-Harmonics Ratio, which measures the ratio of noise to harmonics in the voice signal. A higher ratio can indicate a more disorderly voice.

16. **HNR**: Harmonics-to-Noise Ratio, which measures the ratio of harmonics to noise in the voice signal.

17. **RPDE**: Recurrence Period Density Entropy, a measure of the complexity of the voice signal's periodicity.

18. **DFA**: Detrended Fluctuation Analysis, a measure of the voice signal’s long-range correlations.

19. **spread1**: The ratio of the standard deviation to the mean of the pitch periods. It provides information on pitch variability.

20. **spread2**: The ratio of the interquartile range to the median of the pitch periods. It reflects the dispersion of pitch values.

21. **D2**: The mean of the absolute values of the second derivative of the voice signal. It measures changes in the rate of change of the signal.

22. **PPE**: Pitch Period Entropy, which measures the variability in pitch periods relative to the fundamental frequency.

## Requirements

- Python 3.x
- `numpy`
- `pandas`
- `scikit-learn`
- `librosa`
- `Flask`
- `joblib`
- `Werkzeug`
- `pydub`

You can install the required packages using `pip`:

```bash
pip install numpy pandas scikit-learn librosa Flask joblib Werkzeug pydub
