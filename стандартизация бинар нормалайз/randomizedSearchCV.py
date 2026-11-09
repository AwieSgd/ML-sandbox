import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import (
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# 1. Создаем наш собственный датасет
data = {
    'Hours_Studied': [
        10,
        2,
        np.nan,
        15,
        8,
        3,
        12,
        np.nan,
        6,
        14,
    ],  # Часы учебы (числа)
    'Attendance': [
        95,
        50,
        80,
        np.nan,
        90,
        40,
        85,
        60,
        75,
        100,
    ],  # Посещаемость в % (числа)
    'Major': [
        'IT',
        'Economics',
        'IT',
        'Law',
        np.nan,
        'Law',
        'Economics',
        'IT',
        'Law',
        'IT',
    ],  # Специальность (текст)
    'Has_Scholarship': [
        'Yes',
        'No',
        'Yes',
        'No',
        'No',
        np.nan,
        'Yes',
        'No',
        'Yes',
        'Yes',
    ],  # Стипендия (текст)
    'Passed': [1, 0, 1, 1, 1, 0, 1, 0, 0, 1],  # Цель: сдал (1) или нет (0)
}

df = pd.DataFrame(data)

x = df.drop(columns=['Passed'])
y = df['Passed']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

num_orig = ['Hours_Studied','Attendance']
info_orig = ['Has_Scholarship','Major']

num_set = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
info_set = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                    ('OneHotEncoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_set, num_orig),
    ('info', info_set, info_orig),
])

model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                        ('model', RandomForestClassifier(random_state=42))])

param_random = {
    'model__n_estimators': [10, 20, 50, 100],
    'model__max_depth': [2, 4, 6, 8]
}
model_research = RandomizedSearchCV(
    estimator=model_pipeline,
    param_distributions=param_random,
    n_iter=4,  # Выбираем только 4 случайные комбинации
    cv=3,
    scoring='accuracy',
    random_state=42,
)
model_research.fit(x_train, y_train)

print('--- RANDOMIZED SEARCH ---')
print('Лучшие случайные параметры:', model_research.best_params_)
print('Лучшая точность на кросс-валидации:', model_research.best_score_)

# Проверка на тестовых данных
predictions = model_research.predict(x_test)

print('\nОтчет по метрикам:')
print(classification_report(y_test, predictions, zero_division=0))


