import numpy as np
import pandas as pd
from PIL.ImageChops import screen
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer


data = {
    'MonthlyIncome': [50000, 120000, np.nan, 80000, 45000, 150000, np.nan, 60000, 95000, 70000],
    'YearsAtCompany': [2, 5, 1, np.nan, 3, 10, 4, 1, 6, np.nan],
    'Department': ['IT', 'Sales', 'IT', 'HR', np.nan, 'IT', 'Sales', 'HR', 'Sales', 'IT'],
    'BusinessTravel': ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel', 'Travel_Rarely', np.nan, 'Non-Travel', 'Travel_Frequently', 'Travel_Rarely', 'Non-Travel', 'Travel_Frequently'],
    'Left': [0, 0, 1, 0, 1, 0, 1, 0, 0, 1]
}

df = pd.DataFrame(data)

x = df.drop('Left', axis=1)
y = df['Left']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

num_orig = ['MonthlyIncome', 'YearsAtCompany']
info_orig = ['BusinessTravel', 'Department']

num_set = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent'))])

info_set = Pipeline(steps=[("imputer", SimpleImputer(strategy='most_frequent')),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore',sparse_output=False))])

col = ColumnTransformer(transformers=[
                        ('num', num_set, num_orig),
                         ("info", info_set, info_orig)
                         ])

model_pipeline = Pipeline(steps=[('preprocessor', col),
                                 ("model", LogisticRegression(random_state=42))])

model_pipeline.fit(x_train, y_train)
predictions = model_pipeline.predict(x_test)
cross_val_score(model_pipeline, x_train, y_train, cv=3, scoring='accuracy')
score = model_pipeline.score(x_test, y_test)
print(classification_report(y_test, predictions), '\n')
print(predictions[:5], '\n')
print(score,'\n')
print(y_test[:5])










