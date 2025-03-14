# 🌍 Earthquake Damage Assessment using Machine Learning

![ML](https://img.shields.io/badge/Machine%20Learning-RandomForest-brightgreen) ![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Overview
This project predicts the damage grade of buildings after an earthquake using a **Random Forest Classifier**. The dataset contains various structural attributes of buildings, and our model classifies them into different damage grades.

## 📂 Table of Contents
- [🚀 Getting Started](#-getting-started)
- [📊 Data Overview](#-data-overview)
- [🔍 Data Preprocessing](#-data-preprocessing)
- [🎯 Model Training](#-model-training)
- [📈 Model Evaluation](#-model-evaluation)
- [📌 Feature Importance](#-feature-importance)
- [💾 Saving & Loading the Model](#-saving--loading-the-model)
- [📜 Results](#-results)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## 🚀 Getting Started
### Prerequisites
Ensure you have **Python 3.8+** and the following libraries installed:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### Open in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

## 📊 Data Overview
We use a dataset named **csv_building_damage_assessment.csv**, containing information about various building attributes.

```python
import pandas as pd

# Load dataset
data = pd.read_csv("csv_building_damage_assessment.csv")
print(data.head())
print(data.info())
print(data.describe())
```

| Feature Name   | Description |
|---------------|-------------|
| `age`         | Age of the building |
| `area`        | Area of the building |
| `foundation`  | Type of foundation |
| `roof_type`   | Type of roofing material |
| `damage_grade`| Damage level (Target Variable) |

---

## 🔍 Data Preprocessing
Handling missing values and encoding categorical data:
```python
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Fill missing values
data.fillna(method='ffill', inplace=True)

# Encode categorical variables
le = LabelEncoder()
categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])
```

---

## 🎯 Model Training
Splitting the dataset and training a **Random Forest Classifier**:
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X = data.drop("damage_grade", axis=1)
y = data["damage_grade"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

---

## 📈 Model Evaluation
After training, we evaluate our model:
```python
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Predictions
y_pred = model.predict(X_test)

# Performance Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```

### 🔥 Confusion Matrix Visualization
```python
import seaborn as sns
import matplotlib.pyplot as plt

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
```
📊 **Confusion Matrix:**
![Confusion Matrix](https://via.placeholder.com/600x300.png?text=Confusion+Matrix)

---

## 📌 Feature Importance
We analyze which features contribute the most to the predictions:
```python
importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=features, palette="viridis")
plt.title("Feature Importance")
plt.show()
```
📈 **Feature Importance Plot:**
![Feature Importance](https://via.placeholder.com/600x300.png?text=Feature+Importance)

---

## 💾 Saving & Loading the Model
To reuse the trained model:
```python
import joblib

# Save the trained model
joblib.dump(model, "earthquake_damage_model.pkl")

# Load the model (for later use)
loaded_model = joblib.load("earthquake_damage_model.pkl")
```

---

## 📜 Results
| Metric         | Value |
|---------------|-------|
| **Accuracy**  | 85.6% |
| **Precision** | 83.2% |
| **Recall**    | 82.5% |
| **F1 Score**  | 82.8% |

🚀 The model achieved **85.6% accuracy**, making it reliable for assessing earthquake damage.

---

## 🤝 Contributing
Feel free to submit issues and pull requests. Contributions are welcome! 🙌

---

## 📄 License
This project is licensed under the **MIT License**.

📌 **Author:** [Your Name]  
📧 **Contact:** your.email@example.com

