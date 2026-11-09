from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
data = {
    'Pclass': [3, 1, 3, 1, 3, 2, 3, 1, 2, 3],
    'Sex': [
        'male',
        'female',
        'female',
        'female',
        'male',
        'male',
        'female',
        'male',
        'female',
        'male',
    ],
    'Age': [
        22.0,
        38.0,
        26.0,
        35.0,
        np.nan,
        54.0,
        2.0,
        27.0,
        14.0,
        np.nan,
    ],
    'Fare': [7.25, 71.28, 7.92, 53.1, 8.05, 51.86, 21.07, 30.5, 30.07, 31.27],
    'Survived': [0, 1, 1, 1, 0, 0, 0, 0, 1, 0],
}
df = pd.DataFrame(data)

# Разделяем на признаки (X) и целевую переменную (y)
X = df.drop('Survived', axis=1)
y = df['Survived']

numbers_orig = ['Pclass', 'Age', 'Fare']
info_orig = ['Sex']

numbers_set = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))]) # imputer - это просто имя шага


info_set = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Первый шаг
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # Второй шаг
])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numbers_set, numbers_orig),    # num, words - мы придумали сами, любое название подойдет
        ('words', info_set, info_orig)      ])       # numbers_set и info_set - настройка, которые идут в orig

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestRegressor(n_estimators=100,random_state=42))])
model_pipeline.fit(X, y)
predictions = model_pipeline.predict(X)
print(predictions)






