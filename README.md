# Predictive Maintenance for Manufacturing Equipment

![Predictive Maintenance](https://img.shields.io/badge/Predictive-Maintenance-blue)

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset](#dataset)
- [Modeling and Algorithms](#modeling-and-algorithms)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

This project aims to predict the type of failure in industrial manufacturing equipment based on sensor readings and machine operational data. By predicting failures ahead of time, maintenance can be scheduled proactively to minimize downtime, reduce costs, and prevent catastrophic failures.

The dataset contains sensor readings and operational parameters recorded from industrial machinery over time. Each entry represents a snapshot of the equipment's state at a particular moment. The target is to classify different types of potential failures.

GitHub Repository: [Predictive Maintenance](https://github.com/jaysri125278/Predictive-Maintenance/tree/main)

## Installation

Clone the repository
```bash
git clone https://github.com/jaysri125278/Predictive-Maintenance.git
cd Predictive-Maintenance


Use venv or conda to create a virtual environment and install dependencies.

```bash
# Using venv
python3 -m venv venv
source venv/bin/activate
bash

# Using conda
conda create --name predictive_maintenance python=3.8
conda activate predictive_maintenance
Install dependencies
bash
Copy code
pip install -r requirements.txt
Dataset
The data used in this project is fetched from a MongoDB collection called PredictiveMaintainence. The dataset consists of sensor readings for various machinery, operating conditions, and labels for failure types. Preprocessing includes handling missing values, feature extraction, and balancing the dataset using SMOTENC.

Modeling and Algorithms
The project employs multiple machine learning models, including:

Random Forest Classifier: A robust model for classification tasks.
XGBoost: An optimized gradient boosting algorithm.
Support Vector Machine (SVM): A powerful classifier for high-dimensional spaces.
Multilayer Perceptron (MLP): A deep learning model for complex patterns.
KMeans Clustering: For unsupervised learning and clustering analysis.
To deal with the imbalance in failure types, SMOTENC is used for synthetic oversampling of the minority classes. Dimensionality reduction is done using PCA to improve model performance and interpretability.

Evaluation Metrics
Accuracy
ROC-AUC Score
F1 Score
Confusion Matrix
Silhouette Score for clustering evaluation
Usage
Preprocessing: Clean and preprocess the data using the provided MongoDB dataset.

bash
Copy code
python src/data_preprocessing.py
Train Models: Train various machine learning models.

bash
Copy code
python src/train_model.py
Evaluate: Run evaluation metrics and generate results.

bash
Copy code
python src/evaluate.py
Cluster Analysis: Use clustering for anomaly detection or further insights.

bash
Copy code
python src/clustering_analysis.py
Results
Model Performance: Detailed evaluation metrics such as accuracy, F1 score, and ROC-AUC score are reported for different models.
Clustering: KMeans clustering results are provided for pattern detection and anomaly detection.
The results are stored in the /results folder.
Contributing
Contributions are welcome! Fork the repository, make your changes, and submit a pull request.

Fork the repository
Create a branch for your feature
Implement your feature
Submit a pull request
License
This project is licensed under the MIT License. See the LICENSE file for more information.

Contact
For questions or suggestions, please reach out to:

Name: Your Name
Email: your.email@example.com
