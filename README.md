# 🌸 Iris Flower Classification using Machine Learning

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live_App-red?logo=streamlit)](https://your-streamlit-link-here)

A Machine Learning project that classifies Iris flower species using a K-Nearest Neighbors (KNN) model deployed with Streamlit.

This project demonstrates a complete end-to-end ML workflow including:
- Data loading
- Exploratory Data Analysis (EDA)
- Data preprocessing
- Model training
- Pipeline creation
- Model evaluation
- Model deployment with Streamlit

---

# 👨‍💻 Author

**Prianshu Kumar**

GitHub:  
https://github.com/prianshu-kumar

---

# 🚀 Project Overview

This repository demonstrates a complete machine learning classification workflow using the famous Iris flower dataset.

The model predicts the species of an Iris flower based on:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The target classes are:
- Iris Setosa
- Iris Versicolor
- Iris Virginica

---

# 📂 Dataset

Dataset used:
- **Iris Dataset** from `sklearn.datasets`

The dataset contains:
- 150 flower samples
- 4 numerical features
- 3 flower classes

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Jupyter Notebook

---

# 📊 Exploratory Data Analysis

Performed:
- Pairplots
- Correlation analysis
- Feature distribution visualization
- Decision boundary visualization

Key observations:
- Petal dimensions strongly separate flower species
- Setosa is linearly separable from the other classes

---

# 🤖 Model Pipeline

The project uses a Scikit-learn Pipeline containing:

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("model", KNeighborsClassifier(n_neighbors=5))
])
```

Pipeline components:
- `StandardScaler` → Feature normalization
- `KNeighborsClassifier` → Classification model

---

# 📈 Model Evaluation

Evaluation metrics used:
- Accuracy Score
- Confusion Matrix
- Classification Report

The model achieved high classification accuracy on the Iris dataset.

---

# 🌐 Streamlit Application

The project includes a fully interactive Streamlit web application for real-time predictions.

Users can:
- Enter flower measurements
- Predict Iris species instantly
- Interact with the trained ML model

Run locally using:

```bash
streamlit run app/app.py
```

---

# 📁 Repository Structure

```bash
Iris-Flower-Classification/
│
├── app/
│   └── app.py
│
├── model/
│   └── model.pkl
│
├── notebook/
│   └── main.ipynb
│
├── README.md
└── requirements.txt
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/your-repo-name.git
```

Move into the project directory:

```bash
cd Iris-Flower-Classification
```

Create virtual environment:

```bash
python -m venv venv
```

Activate environment:

### macOS/Linux
```bash
source venv/bin/activate
```

### Windows
```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Usage

## Run Streamlit App

```bash
streamlit run app/app.py
```

## Run Jupyter Notebook

Open:

```bash
notebook/main.ipynb
```

to explore:
- EDA
- Visualizations
- Model training
- Pipeline creation
- Model saving

---

# 💾 Model Export

The trained pipeline model is saved as:

```bash
model/model.pkl
```

using Python Pickle for deployment and inference.

---

# 🔮 Future Improvements

- Add more visualization features
- Deploy using Docker
- Add multiple ML model comparisons
- Hyperparameter tuning
- Improve UI/UX of Streamlit app

---

# 📜 License

This project is created for educational and portfolio purposes.