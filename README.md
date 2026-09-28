# 🤖 AutoML Platform

A full-stack **Automated Machine Learning (AutoML) platform** that automates the machine learning workflow from dataset analysis and preprocessing to model selection, hyperparameter tuning, model evaluation, and report generation.

The platform consists of:

- **Python + FastAPI** backend for the ML pipeline and REST APIs
- **Angular** frontend for user interaction and visualization
- **Scikit-learn** for machine learning workflows
- **Ollama + Qwen 3:8B** for local LLM-powered AI functionality

---

## 🚀 Features

### Dataset Analysis

- Dataset upload
- Automatic dataset profiling
- Row and column analysis
- Data type detection
- Numerical feature detection
- Categorical feature detection
- Boolean feature detection
- Datetime feature detection
- Missing-value analysis
- Duplicate-row detection
- Statistical analysis
- Correlation analysis
- Skewness analysis
- Variance analysis
- Constant-column detection
- Cardinality analysis

### Problem Detection

Automatically identifies the type of machine learning problem based on the dataset and target column.

Supported learning types:

```text
Supervised Learning
├── Regression
└── Classification

Unsupervised Learning
└── Clustering
```

### Preprocessing Recommendation

The system analyzes the dataset and recommends preprocessing operations such as:

- Missing-value handling
- Duplicate handling
- Categorical encoding
- Feature scaling
- Outlier handling
- Feature preprocessing

### Automated Model Selection

The recommendation engine analyzes dataset characteristics and generates suitable candidate models.

Factors considered include:

- Problem type
- Dataset size
- Linearity
- Noise level
- Dimensionality
- Class imbalance
- Feature characteristics

### Model Training

The system automatically trains selected candidate models and generates predictions or cluster assignments.

### Hyperparameter Tuning

The framework supports automated hyperparameter optimization to identify suitable model configurations.

### Cross Validation

Candidate models can be evaluated using cross-validation to obtain more reliable performance measurements.

### Model Evaluation

Different metrics are selected according to the machine learning problem.

#### Regression

- R² Score
- RMSE
- MAE
- MSE

#### Classification

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

#### Clustering

- Silhouette Score
- Calinski-Harabasz Score
- Davies-Bouldin Score

### Report Generation

The backend generates structured analysis and model evaluation results that can be consumed by the frontend.

---

# 🏗️ System Architecture

```text
                         User
                          │
                          ▼
                ┌──────────────────┐
                │ Angular Frontend │
                └────────┬─────────┘
                         │
                         │ HTTP / REST
                         ▼
                ┌──────────────────┐
                │ FastAPI Backend  │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Dataset Upload   │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Dataset Analyzer │
                └────────┬─────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
     Dataset Info   Problem Info   Feature Info
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                ┌──────────────────────┐
                │ Preprocessing Engine │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Recommendation Engine│
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Candidate Models     │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Model Training       │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Cross Validation     │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Hyperparameter       │
                │ Tuning               │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Model Evaluation     │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Report Generation    │
                └──────────────────────┘
```

---

# 🤖 Local LLM Integration

The platform also supports local LLM functionality using:

```text
Ollama + Qwen 3:8B
```

The LLM runs locally instead of requiring an external cloud-based LLM API.

This allows AI-powered functionality to be integrated while keeping the LLM execution on the local machine.

---

## Ollama Setup

### 1. Install Ollama

Download and install Ollama from:

```text
https://ollama.com/download
```

After installation, verify it:

```bash
ollama --version
```

---

### 2. Pull Qwen 3:8B

Pull the Qwen model:

```bash
ollama pull qwen3:8b
```

Check installed models:

```bash
ollama list
```

You should see:

```text
qwen3:8b
```

---

### 3. Run Qwen

Start the model:

```bash
ollama run qwen3:8b
```

You can test the model directly:

```text
>>> Explain the AutoML workflow
```

---

## Ollama API

Ollama exposes a local API by default at:

```text
http://localhost:11434
```

Example API request:

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "qwen3:8b",
  "prompt": "Explain automated model selection in AutoML",
  "stream": false
}'
```

---

## Python Integration with Ollama

The FastAPI backend can communicate with the local Qwen model through the Ollama API.

Example:

```python
import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "qwen3:8b",
        "prompt": "Explain the AutoML pipeline",
        "stream": False
    }
)

result = response.json()

print(result["response"])
```

---

## Local AI Architecture

```text
                 Angular Frontend
                        │
                        ▼
                  FastAPI Backend
                        │
                        ▼
                    AI Service
                        │
                        │ HTTP
                        ▼
                  Ollama Server
                        │
                        ▼
                   Qwen 3:8B
                        │
                        ▼
                AI Generated Output
```

---

# 🛠️ Technology Stack

## Backend

- Python
- FastAPI
- Uvicorn
- Pandas
- NumPy
- Scikit-learn

## Frontend

- Angular
- TypeScript
- HTML
- CSS

## Machine Learning

- Regression
- Classification
- Clustering
- Model selection
- Model recommendation
- Hyperparameter tuning
- Cross-validation
- Model evaluation

## Generative AI

- Qwen 3:8B
- Ollama
- Local LLM API

---

# 📁 Project Structure

```text
AUTOML/
│
├── Backend/
│   │
│   ├── ai_report/
│   ├── dataset/
│   ├── helper/
│   ├── modelTraining/
│   ├── preprocessing/
│   ├── recommendation/
│   ├── reports/
│   ├── routes/
│   ├── services/
│   ├── storage/
│   ├── uploads/
│   │
│   ├── app.py
│   ├── main.py
│   ├── requirements.txt
│   ├── response.json
│   └── sample_report.json
│
├── Frontend/
│   └── automl-schematic-angular/
│       │
│       ├── src/
│       ├── angular.json
│       ├── package.json
│       ├── package-lock.json
│       ├── tsconfig.json
│       └── ...
│
├── .gitignore
└── README.md
```

---

# ⚙️ Installation

## Prerequisites

Make sure the following are installed:

- Python 3.x
- Node.js
- npm
- Angular CLI
- Git
- Ollama

Check versions:

```bash
python --version
```

```bash
node --version
```

```bash
npm --version
```

```bash
ng version
```

```bash
ollama --version
```

---

# 📥 Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/automl.git
```

Navigate into the project:

```bash
cd automl
```

---

# 🐍 Backend Setup

Navigate to the backend:

```bash
cd Backend
```

Create a Python virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Start the Backend

From the `Backend` directory:

```bash
uvicorn app:app --reload
```

The backend will run at:

```text
http://127.0.0.1:8000
```

FastAPI Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

Alternative:

```text
http://127.0.0.1:8000/redoc
```

---

# 🅰️ Frontend Setup

Open a new terminal.

Navigate to:

```bash
cd Frontend/automl-schematic-angular
```

Install dependencies:

```bash
npm install
```

Start the Angular development server:

```bash
ng serve
```

The frontend will be available at:

```text
http://localhost:4200
```

---

# ▶️ Running the Complete Application

Start the components in the following order.

### Step 1 — Start Ollama

```bash
ollama serve
```

If Ollama is already running as a background service, this step may not be required.

Verify the model:

```bash
ollama list
```

Make sure:

```text
qwen3:8b
```

is available.

---

### Step 2 — Start FastAPI

```bash
cd Backend
```

Activate the environment:

```bash
venv\Scripts\activate
```

Start the backend:

```bash
uvicorn app:app --reload
```

---

### Step 3 — Start Angular

Open another terminal:

```bash
cd Frontend/automl-schematic-angular
```

Start Angular:

```bash
ng serve
```

---

# 🔄 Complete Application Flow

```text
                         User
                          │
                          ▼
                  Angular Frontend
                          │
                          ▼
                    File Upload
                          │
                          ▼
                  FastAPI Backend
                          │
                          ▼
                  Dataset Analysis
                          │
                          ▼
                Problem Type Detection
                          │
                          ▼
              Preprocessing Recommendation
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
                  Report Generation
                          │
                          ▼
                   Angular UI
```

For AI-powered functionality:

```text
Angular
   │
   ▼
FastAPI
   │
   ▼
AI Service
   │
   ▼
Ollama
   │
   ▼
Qwen 3:8B
```

---

# 📊 Example Evaluation Output

Regression:

```json
{
    "model": "RandomForestRegressor",
    "r2_score": 0.91,
    "rmse": 12500,
    "mae": 8200
}
```

Classification:

```json
{
    "model": "RandomForestClassifier",
    "accuracy": 0.94,
    "precision": 0.93,
    "recall": 0.92,
    "f1_score": 0.925
}
```

These values are examples only. Actual performance depends on the dataset, preprocessing, model configuration, and train/test split.

---

# 🔐 Configuration

If environment variables are required, create a `.env` file inside the backend.

Example:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:8b
```

Do not commit `.env` files to GitHub.

The `.gitignore` should contain:

```gitignore
.env
*.env
```

---

# 📦 Backend Dependencies

Backend dependencies are maintained in:

```text
Backend/requirements.txt
```

Install them using:

```bash
pip install -r requirements.txt
```

If dependencies are added during development, update the file:

```bash
pip freeze > requirements.txt
```

---

# 📦 Frontend Dependencies

Angular dependencies are maintained in:

```text
Frontend/automl-schematic-angular/package.json
```

Install them using:

```bash
npm install
```

---

# 🎯 Project Objective

The objective of this project is to build a modular AutoML platform that reduces the amount of manual work required during machine learning development.

Traditional workflow:

```text
Dataset
   ↓
Manual Analysis
   ↓
Manual Preprocessing
   ↓
Choose Model
   ↓
Train Model
   ↓
Tune Parameters
   ↓
Evaluate
   ↓
Generate Report
```

AutoML workflow:

```text
Dataset
   ↓
Automatic Analysis
   ↓
Preprocessing Recommendation
   ↓
Automatic Model Recommendation
   ↓
Candidate Model Selection
   ↓
Training
   ↓
Cross Validation
   ↓
Hyperparameter Tuning
   ↓
Evaluation
   ↓
Report
```

---

# 📈 Project Highlights

The project demonstrates implementation of:

- Machine learning automation
- Rule-based model recommendation
- Dataset analysis
- Data preprocessing
- Model training
- Model evaluation
- Hyperparameter optimization
- Cross-validation
- REST API development
- FastAPI backend architecture
- Angular frontend development
- Local LLM integration
- Ollama integration
- Qwen 3:8B integration
- Modular software architecture

---

# 👨‍💻 Author

**Ragulsanjeevan**

---

# 📄 License

This project is currently intended for learning, experimentation, and development purposes.
