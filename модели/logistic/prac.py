from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

data = load_breast_cancer()
# Создаем DataFrame из матрицы X и подписываем колонки именами из датасета
df = pd.DataFrame(data.data, columns=data.feature_names)
# Добавляем колонку с ответами
df["target"] = data.target

x = df.drop(["target"], axis=1)
y = df["target"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)

scaler = StandardScaler()
scaled_train = scaler.fit_transform(x_train)
scaled_test = scaler.transform(x_test)

model = LogisticRegression()
model.fit(scaled_train, y_train)

predictions = model.predict(scaled_test)
probability = model.predict_proba(scaled_test)
score = model.score(scaled_test, y_test)
print("Точность модели (Score):", score, "\n")
print("Что угадала модель (Predictions):", predictions[:10], "\n")
print("Реальные ответы (y_test):", y_test[:10], "\n")
print("Вероятности классов:\n", probability[:10])
print(accuracy_score(y_test[:10], predictions[:10]))


