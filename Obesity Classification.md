```python
#=================================================
# 1. IMPORT LIBRARIES
#==================================================

import numpy as np
import pandas as pd
import os
import joblib

!Pip install shap
import shap
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

from xgboost import XGBClassifier

```

    Requirement already satisfied: shap in c:\users\hp\anaconda3\lib\site-packages (0.51.0)
    Requirement already satisfied: numpy>=2 in c:\users\hp\anaconda3\lib\site-packages (from shap) (2.1.3)
    Requirement already satisfied: scipy in c:\users\hp\anaconda3\lib\site-packages (from shap) (1.15.3)
    Requirement already satisfied: scikit-learn in c:\users\hp\anaconda3\lib\site-packages (from shap) (1.6.1)
    Requirement already satisfied: pandas in c:\users\hp\anaconda3\lib\site-packages (from shap) (2.2.3)
    Requirement already satisfied: tqdm>=4.27.0 in c:\users\hp\anaconda3\lib\site-packages (from shap) (4.67.1)
    Requirement already satisfied: packaging>20.9 in c:\users\hp\anaconda3\lib\site-packages (from shap) (24.2)
    Requirement already satisfied: slicer==0.0.8 in c:\users\hp\anaconda3\lib\site-packages (from shap) (0.0.8)
    Requirement already satisfied: numba in c:\users\hp\anaconda3\lib\site-packages (from shap) (0.61.0)
    Requirement already satisfied: llvmlite in c:\users\hp\anaconda3\lib\site-packages (from shap) (0.44.0)
    Requirement already satisfied: cloudpickle in c:\users\hp\anaconda3\lib\site-packages (from shap) (3.0.0)
    Requirement already satisfied: typing-extensions in c:\users\hp\anaconda3\lib\site-packages (from shap) (4.12.2)
    Requirement already satisfied: colorama in c:\users\hp\anaconda3\lib\site-packages (from tqdm>=4.27.0->shap) (0.4.6)
    Requirement already satisfied: python-dateutil>=2.8.2 in c:\users\hp\anaconda3\lib\site-packages (from pandas->shap) (2.9.0.post0)
    Requirement already satisfied: pytz>=2020.1 in c:\users\hp\anaconda3\lib\site-packages (from pandas->shap) (2024.1)
    Requirement already satisfied: tzdata>=2022.7 in c:\users\hp\anaconda3\lib\site-packages (from pandas->shap) (2025.2)
    Requirement already satisfied: six>=1.5 in c:\users\hp\anaconda3\lib\site-packages (from python-dateutil>=2.8.2->pandas->shap) (1.17.0)
    Requirement already satisfied: joblib>=1.2.0 in c:\users\hp\anaconda3\lib\site-packages (from scikit-learn->shap) (1.4.2)
    Requirement already satisfied: threadpoolctl>=3.1.0 in c:\users\hp\anaconda3\lib\site-packages (from scikit-learn->shap) (3.5.0)
    


```python
# ==============
# 2.LOAD DATASET
# ===============

df = pd.read_csv('ObesityDataSet.csv')
```


```python
# ================
# 3. BASIC EDA
# ================

print('shape:',df.shape)

print("Basic infomation:",df.info())

df.head()


```

    shape: (2111, 17)
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2111 entries, 0 to 2110
    Data columns (total 17 columns):
     #   Column                          Non-Null Count  Dtype  
    ---  ------                          --------------  -----  
     0   Gender                          2111 non-null   object 
     1   Age                             2111 non-null   float64
     2   Height                          2111 non-null   float64
     3   Weight                          2111 non-null   float64
     4   family_history_with_overweight  2111 non-null   object 
     5   FAVC                            2111 non-null   object 
     6   FCVC                            2111 non-null   float64
     7   NCP                             2111 non-null   float64
     8   CAEC                            2111 non-null   object 
     9   SMOKE                           2111 non-null   object 
     10  CH2O                            2111 non-null   float64
     11  SCC                             2111 non-null   object 
     12  FAF                             2111 non-null   float64
     13  TUE                             2111 non-null   float64
     14  CALC                            2111 non-null   object 
     15  MTRANS                          2111 non-null   object 
     16  NObeyesdad                      2111 non-null   object 
    dtypes: float64(8), object(9)
    memory usage: 280.5+ KB
    Basic infomation: None
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Age</th>
      <th>Height</th>
      <th>Weight</th>
      <th>family_history_with_overweight</th>
      <th>FAVC</th>
      <th>FCVC</th>
      <th>NCP</th>
      <th>CAEC</th>
      <th>SMOKE</th>
      <th>CH2O</th>
      <th>SCC</th>
      <th>FAF</th>
      <th>TUE</th>
      <th>CALC</th>
      <th>MTRANS</th>
      <th>NObeyesdad</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Female</td>
      <td>21.0</td>
      <td>1.62</td>
      <td>64.0</td>
      <td>yes</td>
      <td>no</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>Sometimes</td>
      <td>no</td>
      <td>2.0</td>
      <td>no</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>no</td>
      <td>Public_Transportation</td>
      <td>Normal_Weight</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Female</td>
      <td>21.0</td>
      <td>1.52</td>
      <td>56.0</td>
      <td>yes</td>
      <td>no</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>Sometimes</td>
      <td>yes</td>
      <td>3.0</td>
      <td>yes</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>Sometimes</td>
      <td>Public_Transportation</td>
      <td>Normal_Weight</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Male</td>
      <td>23.0</td>
      <td>1.80</td>
      <td>77.0</td>
      <td>yes</td>
      <td>no</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>Sometimes</td>
      <td>no</td>
      <td>2.0</td>
      <td>no</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>Frequently</td>
      <td>Public_Transportation</td>
      <td>Normal_Weight</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Male</td>
      <td>27.0</td>
      <td>1.80</td>
      <td>87.0</td>
      <td>no</td>
      <td>no</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>Sometimes</td>
      <td>no</td>
      <td>2.0</td>
      <td>no</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>Frequently</td>
      <td>Walking</td>
      <td>Overweight_Level_I</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Male</td>
      <td>22.0</td>
      <td>1.78</td>
      <td>89.8</td>
      <td>no</td>
      <td>no</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>Sometimes</td>
      <td>no</td>
      <td>2.0</td>
      <td>no</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Sometimes</td>
      <td>Public_Transportation</td>
      <td>Overweight_Level_II</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('NOBEYESDAD VALUE COUNT:')
print(df['NObeyesdad'].value_counts())
```

    NOBEYESDAD VALUE COUNT:
    NObeyesdad
    Obesity_Type_I         351
    Obesity_Type_III       324
    Obesity_Type_II        297
    Overweight_Level_I     290
    Overweight_Level_II    290
    Normal_Weight          287
    Insufficient_Weight    272
    Name: count, dtype: int64
    


```python
df.describe(include = 'all')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Age</th>
      <th>Height</th>
      <th>Weight</th>
      <th>family_history_with_overweight</th>
      <th>FAVC</th>
      <th>FCVC</th>
      <th>NCP</th>
      <th>CAEC</th>
      <th>SMOKE</th>
      <th>CH2O</th>
      <th>SCC</th>
      <th>FAF</th>
      <th>TUE</th>
      <th>CALC</th>
      <th>MTRANS</th>
      <th>NObeyesdad</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2111</td>
      <td>2111.000000</td>
      <td>2111.000000</td>
      <td>2111.000000</td>
      <td>2111</td>
      <td>2111</td>
      <td>2111.000000</td>
      <td>2111.000000</td>
      <td>2111</td>
      <td>2111</td>
      <td>2111.000000</td>
      <td>2111</td>
      <td>2111.000000</td>
      <td>2111.000000</td>
      <td>2111</td>
      <td>2111</td>
      <td>2111</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2</td>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4</td>
      <td>2</td>
      <td>NaN</td>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4</td>
      <td>5</td>
      <td>7</td>
    </tr>
    <tr>
      <th>top</th>
      <td>Male</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>yes</td>
      <td>yes</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Sometimes</td>
      <td>no</td>
      <td>NaN</td>
      <td>no</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Sometimes</td>
      <td>Public_Transportation</td>
      <td>Obesity_Type_I</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1068</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1726</td>
      <td>1866</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1765</td>
      <td>2067</td>
      <td>NaN</td>
      <td>2015</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1401</td>
      <td>1580</td>
      <td>351</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>NaN</td>
      <td>24.312600</td>
      <td>1.701677</td>
      <td>86.586058</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.419043</td>
      <td>2.685628</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.008011</td>
      <td>NaN</td>
      <td>1.010298</td>
      <td>0.657866</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>6.345968</td>
      <td>0.093305</td>
      <td>26.191172</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.533927</td>
      <td>0.778039</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.612953</td>
      <td>NaN</td>
      <td>0.850592</td>
      <td>0.608927</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>NaN</td>
      <td>14.000000</td>
      <td>1.450000</td>
      <td>39.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>NaN</td>
      <td>19.947192</td>
      <td>1.630000</td>
      <td>65.473343</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.000000</td>
      <td>2.658738</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.584812</td>
      <td>NaN</td>
      <td>0.124505</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>NaN</td>
      <td>22.777890</td>
      <td>1.700499</td>
      <td>83.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.385502</td>
      <td>3.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.000000</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>0.625350</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>NaN</td>
      <td>26.000000</td>
      <td>1.768464</td>
      <td>107.430682</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.477420</td>
      <td>NaN</td>
      <td>1.666678</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>NaN</td>
      <td>61.000000</td>
      <td>1.980000</td>
      <td>173.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.000000</td>
      <td>NaN</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
#checking for missing value
df.isnull().sum()
```




    Gender                            0
    Age                               0
    Height                            0
    Weight                            0
    family_history_with_overweight    0
    FAVC                              0
    FCVC                              0
    NCP                               0
    CAEC                              0
    SMOKE                             0
    CH2O                              0
    SCC                               0
    FAF                               0
    TUE                               0
    CALC                              0
    MTRANS                            0
    NObeyesdad                        0
    dtype: int64




```python
# =========================================
# 4. FEATURE ENGINEERING (BMI)
# =========================================
df['BMI'] = df['Weight'] / (df['Height'] ** 2)
```


```python
df.columns
```




    Index(['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
           'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE',
           'CALC', 'MTRANS', 'NObeyesdad', 'BMI'],
          dtype='object')




```python
# ==================
# 5. visualization
# ==================
```


```python
# Set up visual styling
sns.set(style="whitegrid")

```


```python
# Plot distribution of the target variable (NObeyesdad)
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='NObeyesdad',hue ='NObeyesdad', legend=False, palette="viridis")
plt.title("Distribution of Obesity Levels")
plt.xticks(rotation=45)
plt.show()
```


    
![png](output_10_0.png)
    



```python
# Plot distribution of Age, Weight, and Height
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

sns.histplot(data=df, x="Age", kde=True, ax=axes[0], color="skyblue")
axes[0].set_title("Age Distribution")

sns.histplot(data=df, x="Weight", kde=True, ax=axes[1], color="salmon")
axes[1].set_title("Weight Distribution")

sns.histplot(data=df, x="Height", kde=True, ax=axes[2], color="lightgreen")
axes[2].set_title("Height Distribution")

sns.histplot(data=df, x="BMI", kde=True, ax=axes[3], color="gold")
axes[3].set_title("BMI Distribution")

plt.tight_layout()
plt.show()

```


    
![png](output_11_0.png)
    



```python
# Bar plot for categorical features like Gender, SMOKE, and family history
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.countplot(data=df, x="Gender", ax=axes[0], palette="pastel")
axes[0].set_title("Gender Distribution")

sns.countplot(data=df, x="SMOKE", ax=axes[1], palette="pastel")
axes[1].set_title("Smoking Status")

sns.countplot(data=df, x="family_history_with_overweight", ax=axes[2], palette="pastel")
axes[2].set_title("Family History of Overweight")

plt.tight_layout()
plt.show()
```

    C:\Users\hp\AppData\Local\Temp\ipykernel_14924\1351919423.py:4: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.
    
      sns.countplot(data=df, x="Gender", ax=axes[0], palette="pastel")
    C:\Users\hp\AppData\Local\Temp\ipykernel_14924\1351919423.py:7: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.
    
      sns.countplot(data=df, x="SMOKE", ax=axes[1], palette="pastel")
    C:\Users\hp\AppData\Local\Temp\ipykernel_14924\1351919423.py:10: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.
    
      sns.countplot(data=df, x="family_history_with_overweight", ax=axes[2], palette="pastel")
    


    
![png](output_12_1.png)
    



```python
# Correlation Heatmap (pairplot)

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()
```


    
![png](output_13_0.png)
    



```python

# =========================================
# 6. TARGET ENCODING (REQUIRED FOR XGB)
# =========================================

le = LabelEncoder()
df['NObeyesdad'] = le.fit_transform(df['NObeyesdad'])
```


```python

# =========================================
# 7. DEFINE FEATURE SETS
# =========================================

# Base categorical columns
categorical_cols = ['Gender', 'family_history_with_overweight', 'FAVC',
                    'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']

# Numerical setups
num_base = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
num_bmi_only = ['Age', 'BMI', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
num_combined = ['Age', 'Height', 'Weight', 'BMI', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

```


```python

# Target
X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']

```


```python
# =========================================
# 8. TRAIN / TEST SPLIT (FIRST)
# =========================================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```


```python
# =========================================
# 9. FUNCTION: BUILD PIPELINE
# =========================================
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_combined),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ]
)

```


```python
# =========================================
# 10. DEFINE MODELS (LIGHT vs STRONG)
# =========================================
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

lr_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(max_iter=1000, n_jobs=None))
])

xgb_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', XGBClassifier(
        eval_metric='mlogloss',
        random_state=42
    ))
])
```


```python
# =========================================
# 11. CROSS-VALIDATION SETUP
# =========================================
from sklearn.model_selection import StratifiedKFold, cross_validate

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    'accuracy': 'accuracy',
    'f1': 'f1_weighted'
}

def eval_cv(pipe, name):
    scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    return {
        'Model': name,
        'CV_Accuracy': scores['test_accuracy'].mean(),
        'CV_F1': scores['test_f1'].mean()
    }

lr_cv = eval_cv(lr_pipe, "Logistic Regression")
xgb_cv = eval_cv(xgb_pipe, "XGBoost")

import pandas as pd
cv_results = pd.DataFrame([lr_cv, xgb_cv]).round(4)
print(cv_results)

```

                     Model  CV_Accuracy   CV_F1
    0  Logistic Regression       0.9088  0.9079
    1              XGBoost       0.9775  0.9775
    


```python
# =========================================
# 12. SELECT BEST MODEL (BASED ON CV)
# =========================================
best_name = cv_results.sort_values('CV_F1', ascending=False).iloc[0]['Model']
print("Selected model:", best_name)

best_pipe = xgb_pipe if best_name == "XGBoost" else lr_pipe

```

    Selected model: XGBoost
    


```python
final_model = best_pipe
final_model.fit(X_train, y_train)
```




<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-1 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;preprocessor&#x27;,
                 ColumnTransformer(transformers=[(&#x27;num&#x27;, StandardScaler(),
                                                  [&#x27;Age&#x27;, &#x27;Height&#x27;, &#x27;Weight&#x27;,
                                                   &#x27;BMI&#x27;, &#x27;FCVC&#x27;, &#x27;NCP&#x27;, &#x27;CH2O&#x27;,
                                                   &#x27;FAF&#x27;, &#x27;TUE&#x27;]),
                                                 (&#x27;cat&#x27;,
                                                  OneHotEncoder(drop=&#x27;first&#x27;,
                                                                handle_unknown=&#x27;ignore&#x27;),
                                                  [&#x27;Gender&#x27;,
                                                   &#x27;family_history_with_overweight&#x27;,
                                                   &#x27;FAVC&#x27;, &#x27;CAEC&#x27;, &#x27;SMOKE&#x27;,
                                                   &#x27;SCC&#x27;, &#x27;CALC&#x27;,
                                                   &#x27;MTRANS&#x27;])])),
                (&#x27;model&#x27;,
                 XGBClassifier(base_score=None, booster...
                               feature_types=None, feature_weights=None,
                               gamma=None, grow_policy=None,
                               importance_type=None,
                               interaction_constraints=None, learning_rate=None,
                               max_bin=None, max_cat_threshold=None,
                               max_cat_to_onehot=None, max_delta_step=None,
                               max_depth=None, max_leaves=None,
                               min_child_weight=None, missing=nan,
                               monotone_constraints=None, multi_strategy=None,
                               n_estimators=None, n_jobs=None,
                               num_parallel_tree=None, ...))])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" ><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>Pipeline</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.pipeline.Pipeline.html">?<span>Documentation for Pipeline</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>Pipeline(steps=[(&#x27;preprocessor&#x27;,
                 ColumnTransformer(transformers=[(&#x27;num&#x27;, StandardScaler(),
                                                  [&#x27;Age&#x27;, &#x27;Height&#x27;, &#x27;Weight&#x27;,
                                                   &#x27;BMI&#x27;, &#x27;FCVC&#x27;, &#x27;NCP&#x27;, &#x27;CH2O&#x27;,
                                                   &#x27;FAF&#x27;, &#x27;TUE&#x27;]),
                                                 (&#x27;cat&#x27;,
                                                  OneHotEncoder(drop=&#x27;first&#x27;,
                                                                handle_unknown=&#x27;ignore&#x27;),
                                                  [&#x27;Gender&#x27;,
                                                   &#x27;family_history_with_overweight&#x27;,
                                                   &#x27;FAVC&#x27;, &#x27;CAEC&#x27;, &#x27;SMOKE&#x27;,
                                                   &#x27;SCC&#x27;, &#x27;CALC&#x27;,
                                                   &#x27;MTRANS&#x27;])])),
                (&#x27;model&#x27;,
                 XGBClassifier(base_score=None, booster...
                               feature_types=None, feature_weights=None,
                               gamma=None, grow_policy=None,
                               importance_type=None,
                               interaction_constraints=None, learning_rate=None,
                               max_bin=None, max_cat_threshold=None,
                               max_cat_to_onehot=None, max_delta_step=None,
                               max_depth=None, max_leaves=None,
                               min_child_weight=None, missing=nan,
                               monotone_constraints=None, multi_strategy=None,
                               n_estimators=None, n_jobs=None,
                               num_parallel_tree=None, ...))])</pre></div> </div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>preprocessor: ColumnTransformer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.compose.ColumnTransformer.html">?<span>Documentation for preprocessor: ColumnTransformer</span></a></div></label><div class="sk-toggleable__content fitted"><pre>ColumnTransformer(transformers=[(&#x27;num&#x27;, StandardScaler(),
                                 [&#x27;Age&#x27;, &#x27;Height&#x27;, &#x27;Weight&#x27;, &#x27;BMI&#x27;, &#x27;FCVC&#x27;,
                                  &#x27;NCP&#x27;, &#x27;CH2O&#x27;, &#x27;FAF&#x27;, &#x27;TUE&#x27;]),
                                (&#x27;cat&#x27;,
                                 OneHotEncoder(drop=&#x27;first&#x27;,
                                               handle_unknown=&#x27;ignore&#x27;),
                                 [&#x27;Gender&#x27;, &#x27;family_history_with_overweight&#x27;,
                                  &#x27;FAVC&#x27;, &#x27;CAEC&#x27;, &#x27;SMOKE&#x27;, &#x27;SCC&#x27;, &#x27;CALC&#x27;,
                                  &#x27;MTRANS&#x27;])])</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>num</div></div></label><div class="sk-toggleable__content fitted"><pre>[&#x27;Age&#x27;, &#x27;Height&#x27;, &#x27;Weight&#x27;, &#x27;BMI&#x27;, &#x27;FCVC&#x27;, &#x27;NCP&#x27;, &#x27;CH2O&#x27;, &#x27;FAF&#x27;, &#x27;TUE&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" ><label for="sk-estimator-id-4" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>StandardScaler</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.preprocessing.StandardScaler.html">?<span>Documentation for StandardScaler</span></a></div></label><div class="sk-toggleable__content fitted"><pre>StandardScaler()</pre></div> </div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>cat</div></div></label><div class="sk-toggleable__content fitted"><pre>[&#x27;Gender&#x27;, &#x27;family_history_with_overweight&#x27;, &#x27;FAVC&#x27;, &#x27;CAEC&#x27;, &#x27;SMOKE&#x27;, &#x27;SCC&#x27;, &#x27;CALC&#x27;, &#x27;MTRANS&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" ><label for="sk-estimator-id-6" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>OneHotEncoder</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.preprocessing.OneHotEncoder.html">?<span>Documentation for OneHotEncoder</span></a></div></label><div class="sk-toggleable__content fitted"><pre>OneHotEncoder(drop=&#x27;first&#x27;, handle_unknown=&#x27;ignore&#x27;)</pre></div> </div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" ><label for="sk-estimator-id-7" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>XGBClassifier</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://xgboost.readthedocs.io/en/release_3.0.0/python/python_api.html#xgboost.XGBClassifier">?<span>Documentation for XGBClassifier</span></a></div></label><div class="sk-toggleable__content fitted"><pre>XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, device=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric=&#x27;mlogloss&#x27;,
              feature_types=None, feature_weights=None, gamma=None,
              grow_policy=None, importance_type=None,
              interaction_constraints=None, learning_rate=None, max_bin=None,
              max_cat_threshold=None, max_cat_to_onehot=None,
              max_delta_step=None, max_depth=None, max_leaves=None,
              min_child_weight=None, missing=nan, monotone_constraints=None,
              multi_strategy=None, n_estimators=None, n_jobs=None,
              num_parallel_tree=None, ...)</pre></div> </div></div></div></div></div></div>




```python

# =========================================
# 13. FINAL EVALUATION (TEST SET ONCE)
# =========================================
from sklearn.metrics import classification_report, confusion_matrix

y_pred = final_model.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))
```

    
    Classification Report:
    
                  precision    recall  f1-score   support
    
               0       0.98      1.00      0.99        54
               1       0.98      0.95      0.96        58
               2       0.99      1.00      0.99        70
               3       0.98      1.00      0.99        60
               4       1.00      0.98      0.99        65
               5       0.95      0.97      0.96        58
               6       0.98      0.97      0.97        58
    
        accuracy                           0.98       423
       macro avg       0.98      0.98      0.98       423
    weighted avg       0.98      0.98      0.98       423
    
    
    Confusion Matrix:
    
    [[54  0  0  0  0  0  0]
     [ 1 55  0  0  0  2  0]
     [ 0  0 70  0  0  0  0]
     [ 0  0  0 60  0  0  0]
     [ 0  0  0  1 64  0  0]
     [ 0  1  0  0  0 56  1]
     [ 0  0  1  0  0  1 56]]
    


```python
#=====================
# 14. Best Features
#=====================

# Extract model + preprocessor
model = final_model.named_steps['model']
preprocessor = final_model.named_steps['preprocessor']

# Get categorical feature names
cat_features = preprocessor.transformers_[1][1].get_feature_names_out(categorical_cols)

# Combine with numerical features
feature_names = num_combined + list(cat_features)
```


```python
#feature importance (Tree Based)

import pandas as pd

importance = model.feature_importances_

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

print(importance_df.head(10))
```

                                   Feature  Importance
    3                                  BMI    0.406917
    9                          Gender_Male    0.348582
    16                             SCC_yes    0.069409
    13                      CAEC_Sometimes    0.027057
    2                               Weight    0.017240
    5                                  NCP    0.015251
    0                                  Age    0.014211
    18                      CALC_Sometimes    0.014204
    10  family_history_with_overweight_yes    0.011067
    1                               Height    0.010854
    


```python
# Visualize Top features

plt.figure(figsize=(10,6))
sns.barplot(data=importance_df.head(10), x='Importance', y='Feature')
plt.title("Top 10 Feature Importances")
plt.show()
```


    
![png](output_26_0.png)
    



```python
#============================================
# 15. SHAP : How features affects prediction
#=============================================

# Extract transformed data
X_test_transformed = preprocessor.transform(X_test)
```


```python
# Build SHAP explainer
import shap

explainer = shap.Explainer(model)
shap_values = explainer(X_test_transformed)
```


```python
# SHAP summary plot
shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names)
```


    
![png](output_29_0.png)
    



```python
# Explain one prediction

pred_class = final_model.predict(X_test)[0]

shap.plots.waterfall(shap_values[0, :, pred_class])
```


    
![png](output_30_0.png)
    


## SHAP Explainability Insights

SHAP analysis revealed that BMI is the most influential feature in predicting obesity levels, significantly outperforming individual features such as weight and height.

This confirms that combining weight and height into BMI provides a stronger representation of body composition.

Lifestyle factors such as diet (FCVC), physical activity (FAF), and gender also contribute meaningfully to predictions, while variables like age and water intake have minimal influence.

The results demonstrate that obesity classification is driven primarily by body composition and behavioural patterns.



```python

```
