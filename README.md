# Slovak Sign Language Recognition

A machine learning-based system for recognizing Slovak Sign Language hand gestures using computer vision and hand landmark detection.

## Overview

This project explores the application of computer vision and machine learning techniques to Slovak Sign Language recognition. It uses MediaPipe to detect and extract hand landmarks from video frames, which are then processed and used as input for machine learning models.

The project focuses on both static and dynamic hand gesture recognition, investigating different approaches to gesture representation and classification.

## Technologies

- **Python** – main programming language
- **OpenCV** – image and video processing
- **MediaPipe** – hand detection and landmark extraction
- **scikit-learn** – traditional machine learning models
- **NumPy** – numerical operations and data processing

## Implemented Features

- Hand detection and landmark extraction using MediaPipe.
- Processing and normalization of hand landmark coordinates.
- Dataset collection and preparation.
- Static hand gesture classification.
- Dynamic hand gesture recognition using temporal information.
- Real-time gesture recognition using a webcam.

## Machine Learning Approaches

The project explores several approaches to hand gesture classification:

- **K-Nearest Neighbours (KNN)** – distance-based classification.
- **Support Vector Machine (SVM)** – supervised classification.
- **Random Forest** – ensemble-based classification.
- **Gated Recurrent Unit (GRU)** – sequence-based gesture recognition.
- **Dynamic Time Warping (DTW)** – temporal sequence comparison.

## Project Status

This project is under active development as part of a bachelor's thesis.
