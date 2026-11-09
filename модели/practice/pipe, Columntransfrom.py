import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


data = {
    'Age': [25, np.nan, 32, 47, 22, 38, np.nan, 41, 29, 35],
    'Income': [50000, 82000, np.nan, 120000, 45000, 95000, 60000, np.nan, 75000, 88000],
    'City': ['Almaty', 'Astana', 'Almaty', np.nan, 'Shymkent', 'Astana', 'Almaty', 'Shymkent', np.nan, 'Astana'],
    'Device': ['iOS', 'Android', 'iOS', 'Android', np.nan, 'iOS', 'Android', 'iOS', 'Android', np.nan],
    'Bought': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

x = df.drop('Bought', axis=1)
y = df['Bought']

num_orig = ['Age', 'Income']
info_orig = ['City', 'Device']

num_set = Pipeline(
    steps=[('imputer', SimpleImputer(strategy='constant'))
])

info_set = Pipeline(
    steps=[('imputer', SimpleImputer(strategy='most_frequent')),
            ('Onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
           ])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_set, num_orig),
        ('info', info_set, info_orig)
    ])

model_pipeline = Pipeline(
    steps=[('preprocessor', preprocessor),
           ("classifier", RandomForestRegressor(n_estimators=100, random_state=42))
           ])


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)
model_pipeline.fit(x_train, y_train)
predictions = model_pipeline.predict(x_test)
score = model_pipeline.score(x_test, y_test)
print(score)
print(predictions)

# - самое главное разобраться в скобках