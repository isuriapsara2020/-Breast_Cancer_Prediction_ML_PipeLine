# Breast Cancer Prediction Pipeline

## Project Overview
This repository contains a comprehensive Machine Learning pipeline developed to predict breast cancer outcomes based on patient data. The project is split into two primary predictive tasks:
1. **Classification:** Predicting the mortality status of the patient (Alive vs. Dead).
2. **Regression:** Predicting the survival months specifically for deceased patients.

## Dataset
The project utilizes a dataset named `patients_data.csv`. The raw data includes numerical and categorical variables such as Age, Sex, Tumor Stage (T_Stage), Node Stage (N_Stage), Tumor Size, Estrogen/Progesterone Status, and Regional Nodes Examined. 

## Data Preprocessing & Cleaning
Extensive data cleaning and Exploratory Data Analysis (EDA) were performed to ensure model accuracy. Key preprocessing steps include:
* **Feature Selection:** Dropped irrelevant columns (`Patient_ID`, `Month_of_Birth`, `Occupation`).
* **Outlier Handling:** Detected and removed outliers in the `Tumor_Size` feature utilizing the Interquartile Range (IQR) method.
* **Data Correction:** Addressed inconsistencies by converting negative values in `Age` and `Tumor_Size` to absolutes, and clipped patient ages to a realistic range (0-100).
* **Target Standardization:** Converted the `Mortality_Status` variations into a clean binary format (Alive = 1, Dead = 0).
* **Imputation:** Handled missing values by replacing categorical NaNs with the mode and numerical NaNs with the median.
* **Encoding & Scaling:** Applied `LabelEncoder` for ordinal data (Grade), One-Hot Encoding for nominal categoricals, and `StandardScaler` to normalize numerical features.
* **Data Splitting:** Generated two separate processed datasets (`classification_data.csv` and `regression_data.csv`) tailored for the distinct ML tasks.

## Modeling and Evaluation

### 1. Classification (Target: Mortality Status)
Several classification algorithms were trained and evaluated on an 80/20 train-test split:
* **Logistic Regression:** Tuned using `GridSearchCV` (best parameters: C=0.1, penalty='l2', solver='liblinear'), achieving approximately 86% accuracy.
* **Naive Bayes:** Gaussian Naive Bayes classifier evaluated via Confusion Matrix and ROC-AUC.
* **K-Nearest Neighbors (KNN):** Initialized with 5 neighbors, achieving around 83% accuracy.
* **Ensemble Model:** Built a `VotingClassifier` utilizing soft voting to combine the Logistic Regression and KNN models, yielding an accuracy of ~85%.

### 2. Regression (Target: Survival Months)
Using the subset of data for deceased patients, regression models were applied to predict the number of survival months:
* **Decision Tree Regressor:** Evaluated both a fully-grown decision tree and a pruned tree (max_depth=4).
* Performance metrics evaluated include Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R²) scores. 

## Technologies Used
* **Language:** Python 3
* **Libraries:** Pandas, NumPy, Scikit-Learn, Plotly, Seaborn, Matplotlib
* **Environment:** Google Colab