# 🌧️ Rain Prediction using Machine Learning

A complete end-to-end machine learning project that predicts whether it will rain tomorrow based on historical Australian weather data.  
The project covers data preprocessing, feature engineering, model training, evaluation, and deployment using Streamlit.

---

## 📌 Problem Statement
Accurate rainfall prediction is crucial for agriculture, water resource management, and daily planning.  
This project aims to build a binary classification model to predict **RainTomorrow (Yes/No)** using meteorological features.

---

## 📊 Dataset
- Source: Australian Weather Dataset
- Target Variable: `RainTomorrow`
- Type: Binary Classification
- Features include:
  - Temperature
  - Humidity
  - Pressure
  - Wind speed & direction
  - Rainfall
  - Location & date-based attributes

> Note: The raw dataset is excluded from this repository and should be placed locally when running the project.

---

## 🧠 Machine Learning Workflow
1. Data loading & inspection  
2. Missing value analysis and handling  
3. Feature selection and column removal based on missingness and redundancy  
4. Categorical encoding (One-Hot Encoding)  
5. Feature scaling  
6. Train–test split  
7. Model training and evaluation  
8. Model persistence  
9. Web app deployment using Streamlit  

---

## 🛠️ Tech Stack
- **Language:** Python  
- **Libraries:**  
  - NumPy  
  - Pandas  
  - Scikit-learn  
  - Matplotlib  
  - Seaborn  
  - Streamlit  
- **Tools:**  
  - Git & GitHub  
  - Jupyter Notebook  

---

## 📁 Project Structure
