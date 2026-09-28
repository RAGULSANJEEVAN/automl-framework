# AutoML Platform

A full-stack AutoML platform that automates the machine learning workflow from dataset analysis and preprocessing to model selection, hyperparameter tuning, and model evaluation.

The project provides a Python/FastAPI backend for the ML pipeline and an Angular frontend for interacting with and visualizing the AutoML workflow.

---

## 🚀 Features

- Dataset upload and analysis
- Automatic problem type detection
- Numerical and categorical feature identification
- Missing-value analysis
- Duplicate detection
- Statistical analysis
- Correlation analysis
- Skewness and variance analysis
- Constant-column detection
- Feature cardinality analysis
- Automatic preprocessing recommendations
- Baseline model recommendation
- Candidate model selection
- Automated model training
- Hyperparameter tuning
- Cross-validation
- Model evaluation
- Support for supervised and unsupervised learning
- REST APIs using FastAPI
- Angular-based frontend
- Automated report generation

---

## 🏗️ Architecture

```text
                    User
                     │
                     ▼
            Angular Frontend
                     │
                     │ HTTP / REST API
                     ▼
             FastAPI Backend
                     │
                     ▼
             Dataset Upload
                     │
                     ▼
            Dataset Analyzer
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
      Dataset     Problem     Feature
      Analysis    Detection   Analysis
          │          │          │
          └──────────┼──────────┘
                     ▼
          Preprocessing Engine
                     │
                     ▼
          Model Recommendation
                     │
                     ▼
          Candidate Model Selection
                     │
                     ▼
              Model Training
                     │
                     ▼
            Cross Validation
                     │
                     ▼
          Hyperparameter Tuning
                     │
                     ▼
              Model Evaluation
                     │
                     ▼
              Best Model
                     │
                     ▼
             Report Generation