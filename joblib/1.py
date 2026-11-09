import joblib
from joblib import dump, load

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
data = {
    'Message': [
        'Крути слоты и выиграй джекпот в казино Вулкан! Бонус 500% при регистрации',
        'Привет, во сколько сегодня встретимся у торгового центра?',
        'Заработай миллион за вечер в лучшем онлайн казино! Ссылка в профиле',
        'Слушай, ты скинешь домашку по базам данных?',
        'Казино Вулкан дарит фриспины без депозита заходи быстрее',
        'Купи по дороге молоко и хлеб, пожалуйста',
        'Выиграй машину в новом казино! Успей забрать свой бонус',
        'Отличная погода для прогулки в парке',
        'Как прошла встреча с научным руководителем?',
        'Скину отчет по практике завтра утром',
    ],
    'IsSpam': [1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
}
df = pd.DataFrame(data)
#print(df)

x = df[['Message']]
y = df['IsSpam']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
text = 'Message'
text_set = Pipeline(
    steps=[('tfidf',TfidfVectorizer(max_features=100))])

preprocessor = ColumnTransformer(
    transformers=[('text', text_set, text)])

model = Pipeline(
    steps=[('preprocessor', preprocessor),
           ('model', GradientBoostingClassifier(random_state=42))])
model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)
predictions = model.predict(x_test)

dump(model, '/Users/erasyl/Desktop/spam_detector_model.joblib') # Сохранить

loaded_model = joblib.load('spam_detector_model.joblib') # Загрузить

print(accuracy, '\n')
print('Пайплайн с текстом успешно обучен!')
