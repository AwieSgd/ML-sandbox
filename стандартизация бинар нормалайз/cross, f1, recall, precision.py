import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report

data = {
    'Mileage': [
        15000,
        np.nan,
        120000,
        45000,
        89000,
        np.nan,
        34000,
        110000,
        78000,
        150000,
    ],
    'Year': [2021, 2019, 2015, 2022, 2018, 2020, 2023, 2014, 2017, np.nan],
    'Brand': [
        'Toyota',
        'BMW',
        'Toyota',
        np.nan,
        'Hyundai',
        'BMW',
        'Toyota',
        'Hyundai',
        np.nan,
        'BMW',
    ],
    'Fuel_Type': [
        'Petrol',
        'Diesel',
        np.nan,
        'Petrol',
        'Diesel',
        'Petrol',
        np.nan,
        'Diesel',
        'Petrol',
        'Diesel',
    ],
    'Is_Sold': [1, 0, 0, 1, 1, 0, 1, 0, 1, 0],
}

df = pd.DataFrame(data)

x = df.drop(columns=['Is_Sold'])
y = df['Is_Sold']

num_orig = ['Mileage', 'Year']
info_orig = ['Fuel_Type', 'Brand']

num_set = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))])

info_set = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                           ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_set, num_orig),
    ('info', info_set, info_orig),
])

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model_pipeline.fit(x_train, y_train)

score = cross_val_score(model_pipeline, x_train, y_train, cv=3)
predictions1 = model_pipeline.predict(x_test)
classification_report = classification_report(y_test, predictions1)

print(classification_report, '\n')
print(score)


