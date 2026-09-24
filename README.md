<div align="center">

# 🏡 California House Price Prediction
### End-to-End Machine Learning Web Application

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

<p align="center">
  A complete machine learning solution to estimate median housing prices across California block groups using historical census data, data processing pipelines, and interactive deployment.
</p>

[Project Overview](#-project-overview) • [Architecture](#-system-architecture) • [Dataset](#-dataset-overview) • [Setup Instructions](#-quick-start-guide) • [Inference Logging](#-prediction-history)

</div>

---

## 📌 Project Overview

This repository provides a reproducible end-to-end data science pipeline built on the classic California housing census records:
- **Exploratory Data Analysis (EDA):** Feature distribution checks, skewness detection, and correlation analysis conducted inside `data.ipynb`.
- **Data Preprocessing & Scaling:** Feature normalization implemented using `StandardScaler` to ensure numerical balance across feature ranges.
- **Model Persistence:** Fitted scaler saved as `scaler.pkl` to guarantee identical transforms during live inference.
- **Web Deployment:** A lightweight web application (`app.py`) allowing interactive input submission and instant price calculation.
- **Automated Audit Logging:** Inferences are recorded row-by-row into `prediction_history.csv` for post-run inspection.

---

## 🏗 System Architecture

```plaintext
      User Input (Web Interface)
                 │
                 ▼
          [ app.py Entrypoint ]
                 │
                 ├──► 1. Preprocess: scaler.pkl (Transforms Raw Features)
                 ├──► 2. Model Inference: Generates Target Estimate
                 └──► 3. Data Logger: Appends Request to prediction_history.csv
                 │
                 ▼
        Rendered Prediction ($)

📁 Repository Structure

california_house_prediction/
│
├── assets/                  # Application diagrams, plots, and UI screenshots
├── app.py                   # Production web application entry point
├── data.ipynb               # End-to-end model exploration, cleaning, and training
├── prediction_history.csv   # Historical inference log for inputs and results
├── requirements.txt         # Core dependencies and environment specification
├── scaler.pkl               # Serialized StandardScaler pipeline artifact
└── README.md                # Interactive project documentation

