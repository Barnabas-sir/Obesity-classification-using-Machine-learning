# Obesity Classification using Machine Learning

## Overview

This project develops a machine learning model to classify individuals into 7 obesity categories using demographic, physical, and lifestyle features.

The problem is formulated as a **multi-class classification task**.

---

## Dataset

* 2,111 samples
* 17 features
* Target: `NObeyesdad` (7 classes)

Features include:

* Physical: Height, Weight, BMI
* Lifestyle: diet, activity, alcohol consumption
* Demographics: age, gender

---

## Methodology

### 1. Data Preprocessing

* Feature engineering: **BMI** derived from height and weight
* Numerical features scaled using `StandardScaler`
* Categorical variables encoded using `OneHotEncoder`
* Pipeline used to prevent data leakage

---

### 2. Model Comparison

| Model               | CV Accuracy | CV F1  |
| ------------------- | ----------- | ------ |
| Logistic Regression | 0.9088      | 0.9079 |
| XGBoost             | 0.9775      | 0.9775 |

**Insight:**
The strong performance gap indicates that the problem is **non-linear**, making tree-based models more suitable.

---

### 3. Final Model

* Selected model: **XGBoost**
* Achieved ~98% accuracy on test data
* Balanced performance across all classes

---

## Evaluation Results

* Accuracy: **0.98**
* Weighted F1-score: **0.98**

Confusion matrix shows minimal misclassification, mainly between adjacent obesity categories.

---

## Feature Importance

Top contributing features:

* **BMI (dominant predictor)**
* Weight
* Dietary habits (FCVC)
* Gender
* Physical activity (FAF)

BMI significantly improves predictive performance by combining height and weight into a single meaningful metric.

---

## Model Explainability (SHAP)

SHAP (SHapley Additive exPlanations) was used to interpret model predictions.

### Key Insights:

* BMI is the strongest driver of predictions
* Lifestyle factors (diet and activity) significantly influence outcomes
* Some features (e.g. age, water intake) have minimal impact

SHAP provides:

* Global feature importance
* Direction of feature influence
* Individual prediction explanations

---

## Key Findings

* Obesity classification is driven by a combination of:

  * Body composition (BMI, weight)
  * Lifestyle habits (diet, activity)
* Some class overlap exists, limiting perfect classification

---

## Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* SHAP
* Matplotlib, Seaborn

---

## Future Improvements

* Deploy model with Streamlit
* Add SHAP-based interactive explanations
* Explore deep learning approaches

---

## Author

BARNABAS OGODO

<img width="858" height="600" alt="output_30_0" src="https://github.com/user-attachments/assets/8c64d402-ea88-4f59-a4d5-ef401afececb" />
<img width="783" height="934" alt="output_29_0" src="https://github.com/user-attachments/assets/474e9d66-141d-451d-a710-77e0484e4286" />
<img width="1072" height="550" alt="output_26_0" src="https://github.com/user-attachments/assets/129e5122-492a-476f-9253-356cb41933b1" />
<img width="780" height="683" alt="output_13_0" src="https://github.com/user-attachments/assets/ba4ae8eb-53aa-44b4-9766-ac8294dee5d0" />
<img width="1784" height="484" alt="output_12_1" src="https://github.com/user-attachments/assets/13ff31dd-49d8-4967-bd3d-be5142e078de" />
<img width="1984" height="484" alt="output_11_0" src="https://github.com/user-attachments/assets/8e155231-f406-45d5-8659-6c24ba05af83" />
<img width="854" height="644" alt="output_10_0" src="https://github.com/user-attachments/assets/220d9669-788d-4b66-992f-7eb1e20ba0bf" />

