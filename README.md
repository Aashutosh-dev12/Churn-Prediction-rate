🤖 ChurnSight — Customer Churn Prediction System
🚀 Overview

Customer churn is one of the biggest challenges businesses face, directly impacting revenue and growth.

ChurnSight is a machine learning-powered application that predicts whether a customer is likely to leave, enabling businesses to take proactive retention actions instead of reacting after the loss.

This project goes beyond just model building — it delivers a complete end-to-end ML solution, including data preprocessing, model optimization, and a deployed interactive web app.

🎯 Problem Statement

Businesses often struggle to:

Identify customers at risk of leaving
Understand key factors influencing churn
Take timely actions to retain valuable customers

This project addresses these challenges by building a system that predicts churn before it happens.

🧠 Objectives
Build a robust machine learning model for churn prediction
Handle imbalanced data effectively
Optimize model performance using hyperparameter tuning
Deploy a real-time prediction system
Provide actionable insights for business decisions
🛠️ Tech Stack
Python (Pandas, NumPy) → Data preprocessing
Scikit-learn → Model building & evaluation
Random Forest → Core prediction algorithm
SMOTE → Handling class imbalance
GridSearchCV → Hyperparameter tuning
Plotly → Interactive visualizations
Streamlit → Web app deployment
Git & GitHub → Version control

ChurnSight/
│
├── data/                 # Dataset files
├── notebooks/           # EDA & experimentation
├── models/              # Saved trained model
├── app/                 # Streamlit app code
├── utils/               # Helper functions
├── requirements.txt     # Dependencies
└── README.md            # Project documentation

📊 Key Features
🔹 Data Preprocessing
Handled missing values and encoded categorical variables
Scaled numerical features for better model performance
🔹 Handling Imbalanced Data
Applied SMOTE (Synthetic Minority Oversampling Technique)
Balanced dataset to improve prediction reliability
🔹 Model Building
Trained a Random Forest Classifier
Captured complex patterns in customer behavior
🔹 Model Optimization
Used GridSearchCV to find optimal hyperparameters
Improved accuracy and generalization
🔹 Deployment
Built an interactive Streamlit web application
Enabled real-time churn prediction
