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

🚀 Getting Started
Python 3.9 or higher

Git installed on your system

Pip package manager

# Step 1: Clone the Repository

git clone [https://github.com/](https://github.com/)shachidwivedi19/california_house_prediction.git
cd california_house_prediction


# Step 2: Environment Setup

Create virtual environment
python -m venv venv

Activate virtual environment

Windows (PowerShell/CMD):
venv\Scripts\activate

macOS/Linux:
source venv/bin/activate

# Step 3: Install Dependencies
pip install -r requirements.txt

# Step 4: Run the App
python app.py
Access the application dashboard at http://127.0.0.1:5000 (or http://localhost:8501).

# 📝 Prediction Logger
Every inference request passes through the application layer and commits an immutable record into prediction_history.csv:
MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Latitude,Longitude,PredictedPrice
8.3252,41.0,6.984,1.023,322.0,2.555,37.88,-122.23,452600.00

