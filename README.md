# Project-006-House-price-predictor-full-pipeline-with-deployed-notebook
---
---

## 📘 Project Overview
The **House Price Predictor** project uses machine learning to predict the selling price of a house based on its features such as lot size, number of rooms, quality, year built, and more.  
This project demonstrates a **complete end-to-end ML workflow** — from data preprocessing to model tuning, evaluation, and deployment.

---

## 🧭 Workflow Summary

### **1. Data Collection & Inspection**
- Dataset loaded using `pandas`.
- Performed basic checks: missing values, data types, and descriptive statistics.

### **2. Data Preprocessing**
- **Numeric Columns**: Scaled using `StandardScaler`.
- **Categorical Columns**: Encoded with `OneHotEncoder`.
- Combined into a unified preprocessing pipeline with `ColumnTransformer`.

### **3. Train-Test Split**
- Data split into **80% training** and **20% testing** using `train_test_split`.

### **4. Model Training**
Three models were trained and compared:
- **Linear Regression** (Baseline)
- **Random Forest Regressor**
- **XGBoost Regressor**

### **5. Hyperparameter Tuning**
- Used `GridSearchCV` to find the best hyperparameters for Random Forest and XGBoost.
- Optimized for **R² score** using 3-fold cross-validation.

### **6. Model Evaluation**
Evaluated on the test set using:
- **R² (Coefficient of Determination)**
- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Squared Error)**

Example results:

| Model | R² | MAE | RMSE |
|--------|----|-----|------|
| Linear Regression | 0.82 | 21,000 | 32,000 |
| Random Forest | 0.89 | 16,000 | 25,000 |
| XGBoost | **0.91** | **14,500** | **22,800** |

✅ **XGBoost performed best overall**.

---

## 💾 Model Saving & Loading
The final model was serialized with **Joblib** for deployment.

```python
# Save the best model
import joblib
joblib.dump(best_model, "house_price_predictor.pkl")

# Load it later
model = joblib.load("house_price_predictor.pkl")
pred = model.predict(new_data)

---

## 🧮 Deployment Notebook

An end-to-end Jupyter Notebook was created with:

* Step-by-step training pipeline
* Evaluation metrics
* Model saving/loading
* Example prediction with user inputs

You can run this notebook on **Google Colab**, **Kaggle**, or **JupyterLab**.

---

## 🧠 Key Learnings

* How to build a **full ML pipeline** using `Pipeline` and `ColumnTransformer`.
* Difference between **Linear Regression**, **Random Forest**, and **XGBoost**.
* Importance of **Hyperparameter Tuning** using `GridSearchCV`.
* Saving and reusing trained models efficiently with **Joblib**.
* Structuring a **deployment-ready ML notebook**.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**, **NumPy**
* **Scikit-learn**
* **XGBoost**
* **Joblib**
* **Matplotlib / Seaborn** *(for optional visualizations)*

---

## 📈 Future Improvements

* Add feature importance visualization.
* Include model monitoring in a deployed environment.
* Build a Streamlit app for live user prediction.

---

## 👨‍💻 Author

**Olawale Samuel Olaitan**
AI & ML Enthusiast | Aspiring AI Engineer
🔗 [GitHub](https://github.com/Olanle) • [LinkedIn](https://www.linkedin.com/in/olawalesamuelolaitan)

---

```
