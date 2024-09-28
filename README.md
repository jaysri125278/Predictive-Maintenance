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

1. Clone this repository to your local machine.
2. Install the required dependencies by running `pip install -r requirements.txt`.
3. Set up a MongoDB database and configure the connection details in the project.

### Dataset

The dataset consists of 10 000 data points stored as rows with 14 features in columns
- **UID**: unique identifier ranging from 1 to 10000
- **productID**: consisting of a letter L, M, or H for low (50% of all products), medium (30%), and high (20%) as product quality variants and a variant-specific serial number
- **air temperature [K]**: generated using a random walk process later normalized to a standard deviation of 2 K around 300 K
- **process temperature [K]**: generated using a random walk process normalized to a standard deviation of 1 K, added to the air temperature plus 10 K.
- **rotational speed [rpm]**: calculated from power of 2860 W, overlaid with a normally distributed noise
- **torque [Nm]**: torque values are normally distributed around 40 Nm with an Ïƒ = 10 Nm and no negative values.
- **tool wear [min]**: The quality variants H/M/L add 5/3/2 minutes of tool wear to the used tool in the process. and a 'Machine failure' label that indicates whether the machine has failed in this particular data point for any of the following failure modes is true.
- **Target** : Failure or Not, Failure Type : Type of Failure (There are two targets dont take these as features

### Technologies
- python
- MongoDB
- Jupyter Notebook

## Python Libraries
- **Pandas & NumPy** – Data manipulation and numerical computing.
- **Matplotlib & Seaborn** – Data visualization.
- **Scikit-learn** – For preprocessing, model training, and evaluation.
- **XGBoost** – High-performance classifier.
- **Imbalanced-learn (SMOTENC)** – Handling imbalanced data.
- **PyMong**o – MongoDB integration.
- **Pickle** – Saving/loading models.
- **Isolation Forest** – Anomaly detection.

### Modeling and Algorithms
The project employs multiple machine learning models, including:

- **Random Forest Classifier**: A robust model for classification tasks.
- **XGBoost**: An optimized gradient boosting algorithm.
- **Multilayer Perceptron (MLP)**: A deep learning model for complex patterns.
- **KMeans Clustering**: For unsupervised learning and clustering analysis.

To deal with the imbalance in failure types, SMOTENC is used for synthetic oversampling of the minority classes. Dimensionality reduction is done using PCA to improve model performance and interpretability.

### Evaluation Metrics
- Accuracy
- ROC-AUC Score
- F1 Score
- Confusion Matrix
- Silhouette Score for clustering evaluation

### Contact
For questions or suggestions, please reach out to:

**Name**: Jaysri Saravanan
**Email**: saravananjaysri@gmail.com




