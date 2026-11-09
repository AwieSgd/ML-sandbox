import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer  # <--- Вот он!

data = {
    'MonthlyIncome': [50000, 120000, 80000, 45000, 150000, 60000],
    'Age': [25, 34, 29, 22, 48, 24],
    'Department': ['IT', 'Sales', 'HR', 'IT', 'Sales', 'HR'],
    # Добавили текстовую колонку с отзывами
    'Feedback': [
        'хочу уволиться',
        'все отлично зарплата супер',
        'мало денег и много работы',
        'люблю компанию',
        'планирую уходить',
        'хороший коллектив',
    ],
    'Left': [1, 0, 1, 0, 1, 0],
}

df = pd.DataFrame(data)

x = df.drop(columns='Left')
y = df['Left']

# Списки колонок по типам
num_cols = ['MonthlyIncome', 'Age']
info_cols = ['Department']
text_cols = 'Feedback'  # Текст обрабатывается отдельно (передается как строка, не список!)

# Препроцессоры для каждого типа данных
num_set = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])

info_set = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        (
            'onehot',
            OneHotEncoder(handle_unknown='ignore', sparse_output=False),
        ),
    ]
)

# Текстовый препроцессор на базе TF-IDF
text_set = Pipeline(
    steps=[
        (
            'tfidf',
            TfidfVectorizer(max_features=100),
        ),
    ]
)

# Собираем всё вместе в ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_set, num_cols),
        ('info', info_set, info_cols),
        ('text', text_set, text_cols),  # <--- Добавили обработку текста
    ]
)

# И финальный пайплайн с бустингом
model_pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('model', GradientBoostingClassifier(random_state=42)),
    ]
)

# Обучаем модель на таблице, где есть текст, числа и категории!
model_pipeline.fit(x, y)

print('Пайплайн с текстом успешно обучен!')