from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# 1. Загрузка данных
iris = load_iris()
print(iris.feature_names, "\n")
print(iris.target_names)
print(iris.data.shape, "\n")

X = iris.data
y = iris.target

# 2. Разделение на тренировку и тест
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Обязательный скейлинг данных
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# 4. Цикл для поиска лучшего K (от 1 до 11)
k_range = range(1, 12)
best_score = 0
best_k = 1

for k in k_range:
  knn = KNeighborsClassifier(n_neighbors=k)
  knn.fit(x_train_scaled, y_train)
  predictions = knn.predict(x_test_scaled)
  score = accuracy_score(y_test, predictions)

  if score > best_score:
    best_score = score
    best_k = k

print(f"Победитель! Лучшее количество соседей (k): {best_k}")
print(f"Точность при этом: {best_score}\n")

# 5. Обучаем финальную модель с найденным лучшим K
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(x_train_scaled, y_train)

# 6. Финальные предсказания и вероятности
final_predictions = best_knn.predict(x_test_scaled)
probability = best_knn.predict_proba(x_test_scaled)
accuracy = accuracy_score(y_test, final_predictions)


print("Итоговая точность:", accuracy)