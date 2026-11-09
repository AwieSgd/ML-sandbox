from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
import pandas as pd
import numpy as np
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
}
df = pd.DataFrame(data)

x = df.drop(columns='Left')
y = df['Left']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

num = ['MonthlyIncome', 'YearsAtCompany', 'Age']
info = ["BusinessTravel", "Department"]

num_set = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
info_set = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                           ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

preprocessor = ColumnTransformer(
    transformers=[('num', num_set, num),
                  ('info', info_set, info)])

model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('model', GradientBoostingClassifier(random_state=42))])

param_grid = {
    'model__n_estimators': [50, 100, 150],
    'model__learning_rate': [0.01, 0.1, 0.2],
    'model__max_depth': [3, 5]}

model_research = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy')
model_research.fit(x_train, y_train)
predictions = model_research.predict(x_test)
classi = classification_report(y_test, predictions)

print('--- GRADIENT BOOSTING GRID SEARCH ---')
print(model_research.best_params_, '\n')
print(model_research.best_score_, '\n')
print(classi,'\n')
print(y_test,'\n')
print(predictions)

#Представь, что ты пытаешься поразить мишень из лука, а у тебя в команде целая гильдия лучников.
#Первый стрелок делает выстрел по центру, но сильно мажет и пускает стрелу в левый нижний угол.
#Второй стрелок не пытается поразить всю мишень с нуля — он смотрит на ошибку первого,
#делает поправку («ага, нас сдуло влево и вниз») и стреляет, чтобы компенсировать именно этот промах.
# Но всё равно чуть-чуть не дотягивает.
#Третий стрелок исправляет ошибки второго, четвертый — третьего, и так далее.


