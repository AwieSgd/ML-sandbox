# Regressor - отвечает на вопрос "сколько". Предикт зарплаты.

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

# Датасет, где зарплата (MonthlyIncome) — это то, что мы предсказываем
data = {
    'YearsAtCompany': [2, 5, 1, np.nan, 3, 10, 4, 1, 6, np.nan, 8, 0],
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
    'MonthlyIncome': [
        50000,
        120000,
        65000,
        80000,
        45000,
        150000,
        75000,
        60000,
        95000,
        70000,
        110000,
        35000,
    ],
}

df = pd.DataFrame(data)

# X — все признаки, кроме целевой переменной
x = df.drop(columns=['MonthlyIncome'])
# y — то, что предсказываем (число)
y = df['MonthlyIncome']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

num_orig = ['Age', 'YearsAtCompany']
info_orig = ['BusinessTravel', 'Department', 'Left']

num_set = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
info_set = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        (
            'Onehotencoder',
            OneHotEncoder(handle_unknown='ignore', sparse_output=False),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_set, num_orig),
        ('info', info_set, info_orig),
    ]
)

model_pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('model', DecisionTreeRegressor(random_state=42)),
    ]
)

param_grid = {
    'model__max_depth': [2, 4, 6, 8],
}

model_research = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    cv=3,
    scoring='r2',  # Метрика для регрессии
)

model_research.fit(x_train, y_train)

print('--- GRID SEARCH (REGRESSOR) ---')
print('Лучшие параметры:', model_research.best_params_)
print('Лучший R2 на кросс-валидации:', model_research.best_score_)

# Предсказание точных чисел
predictions = model_research.predict(x_test)

print('\nМетрики на тестовых данных:')
print('MAE (ошибка в прогнозе):', mean_absolute_error(y_test, predictions))
print('R2 score:', r2_score(y_test, predictions))