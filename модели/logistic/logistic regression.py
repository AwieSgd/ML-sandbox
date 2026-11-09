from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. Данные
x = np.array(
    [[2.5, 3, 1.6], [6, 1, 5], [1, 2, 3], [9.1, 5.5, 1], [1.1, 5.9, 4], [0, 4, 4.5]]
)
y = np.array([0, 1, 0, 1, 1, 0])

# 2. Делим на train и test
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.5, random_state=42
)

# 3. Скейлим (обучаем скейлер ТОЛЬКО на train, а test просто трансформируем)
scaler = StandardScaler()
scaled_train = scaler.fit_transform(x_train)
scaled_test = scaler.transform(x_test)  # Переименовал для порядка!

# 4. Создаем и обучаем модель ТОЛЬКО на train
model = LogisticRegression()
model.fit(scaled_train, y_train)

# 5. ДЕЛАЕМ ПРЕДСКАЗАНИЯ НА ТЕСТЕ (на том, что модель НЕ видела)
predictions = model.predict(scaled_test)
probability = model.predict_proba(scaled_test)

# 6. СЧИТАЕМ ОЦЕНКУ (сравниваем тест-предсказания с тест-ответами y_test)
score = model.score(scaled_test, y_test)

print("Точность модели (Score):", score, "\n")
print("Что угадала модель (Predictions):", predictions, "\n")
print("Реальные ответы (y_test):", y_test, "\n")
print("Вероятности классов:\n", probability)
print(accuracy_score(y_test, predictions))