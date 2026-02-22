# Diabetes Risk Prediction Model

A machine learning model trained to predict the risk of diabetes using the PIMA Indians Diabetes Dataset.

This repository focuses on model development, preprocessing, training, evaluation, and serialization.

---

## Overview

This project implements a structured machine learning pipeline including:

- Data preprocessing and missing value handling  
- Feature scaling using StandardScaler  
- Model training using Random Forest Classifier with tuned parameters
- Evaluation using classification metrics and confusion matrix  
- Model serialization for future deployment  

The final trained model and scaler are saved for inference use.

---

## Model Details

- Algorithm: Random Forest Classifier  
- Feature Scaling: StandardScaler  
- Target Variable: Binary classification (0 = No Diabetes, 1 = Diabetes)  
- Evaluation Metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Confusion Matrix  

---

## Project Structure

```
diabetes-risk-predictor/
│
├── train.py
├── requirements.txt
├── README.md
│
├── notebook/
│   └── training.ipynb
│
├── data/
│   └── diabetes.csv
│
└── model/
    ├── diabetes_model.pkl
    └── scaler_dia.pkl
```

---

## Dataset

PIMA Indians Diabetes Dataset  
Public medical dataset containing diagnostic health measurements.

---

## Reproducibility

The model can be retrained by executing:

```
python train.py
```

This will regenerate the serialized model and scaler.

---

## Future Scope

- Model hyperparameter tuning  
- Feature engineering improvements  
- Web or API deployment  
- Model comparison benchmarking  

---

## Author

Shamveel  

