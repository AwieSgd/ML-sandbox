# Сlassifier - отвечает на вопрос "кто и что это". Уйдет сотрудник или нет.
# Regressor - отвечает на вопрос "сколько". Предикт цены и тд.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, classification_report


data = {
    'MonthlyIncome': [
        50000,
        120000,
        np.nan,
        80000,
        45000,
        150000,
        np.nan,
        60000,
        95000,
        70000,
        110000,
        35000,
    ],
    'YearsAtCompany': [
        2,
        5,
        1,
        np.nan,
        3,
        10,
        4,
        1,
        6,
        np.nan,
        8,
        0,
    ],
    'Age': [25, 34, 45, 29, 22, 48, 36, 24, 41, 30, 39, 21],
    'Department': [
        'IT',
        'Sales',
        'IT',
        'HR',
        np.nan,
        'IT',
        'Sales',
        'HR',
        'Sales',
        'IT',
        'HR',
        'Sales',
    ],
    'BusinessTravel': [
        'Travel_Rarely',
        'Travel_Frequently',
        'Non-Travel',
        'Travel_Rarely',
        np.nan,
        'Non-Travel',
        'Travel_Frequently',
        'Travel_Rarely',
        'Non-Travel',
        'Travel_Frequently',
        'Travel_Rarely',
        'Non-Travel',
    ],
    'Left': [0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
}

df = pd.DataFrame(data)
#print(df)

x = df.drop(columns=['Left'])
y = df['Left']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

num_orig = ["Age", "MonthlyIncome", "YearsAtCompany"]
info_orig = ["BusinessTravel", "Department"]

num_set = Pipeline(steps=[("imputer",SimpleImputer(strategy="most_frequent"))])
info_set = Pipeline(steps=[("imputer",SimpleImputer(strategy="most_frequent")),
                           ("Onehotencoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_set, num_orig),
        ("info", info_set, info_orig)])

model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
        ("model", DecisionTreeClassifier(random_state=42))
])

param_grid = {
    "model__max_depth": [2, 4, 6, 8],
}

model_research = GridSearchCV(
    estimator=model_pipeline, param_grid=param_grid, cv=3, scoring='accuracy',
)

model_research.fit(x_train, y_train)

print('--- GRID SEARCH ---')
print('Лучшие параметры:', model_research.best_params_)
print('Лучшая точность на кросс-валидации:', model_research.best_score_)

# 6. Проверка на тестовых данных (grid_search сам стал лучшей моделью!)
predictions = model_research.predict(x_test)

print('\nОтчет по метрикам для Grid Search:')
print(classification_report(y_test, predictions))
