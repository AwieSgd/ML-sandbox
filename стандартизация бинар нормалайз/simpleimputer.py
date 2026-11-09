import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# Создаем объемный и реалистичный датафрейм с пропусками и текстом
data = {
    'PassengerId': range(1, 16),
    'Pclass': [3, 1, 3, 1, 3, 2, 3, 1, 2, 3, 1, 3, 2, 1, 3],
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
        'female',
        'male',
        'female',
        'male',
        'female',
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
        4.0,
        58.0,
        np.nan,
        28.0,
        40.0,
        19.0,
    ],
    'SibSp': [1, 1, 0, 1, 0, 0, 3, 1, 1, 4, 0, 0, 1, 0, 0],
    'Fare': [
        7.25,
        71.28,
        7.92,
        53.1,
        8.05,
        51.86,
        21.07,
        30.5,
        30.07,
        31.27,
        26.55,
        8.05,
        13.0,
        27.72,
        7.87,
    ],
    'Embarked': [
        'S',
        'C',
        'S',
        'S',
        'S',
        'S',
        'S',
        'S',
        'C',
        'Q',
        'S',
        'S',
        'S',
        'C',
        'Q',
    ],
    'Survived': [0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
}

# Специально добавляем несколько пропусков в категориальную колонку Embarked
df = pd.DataFrame(data)
df.loc[4, 'Embarked'] = None
df.loc[10, 'Embarked'] = None

num_input = SimpleImputer(strategy='median')
df[['Age']] = num_input.fit_transform(df[['Age']]) #не забываем про - []

data_input = SimpleImputer(strategy='most_frequent')
df[['Embarked']] = data_input.fit_transform(df[['Embarked']]) #не забываем про - []

print('Размер датафрейма:', df.shape)
print(df)