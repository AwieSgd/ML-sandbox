from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

df = load_breast_cancer(as_frame=True)
#print(df.feature_names)

x = df.data
y = df.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

k_range = range(1,11)
best_score = 0
best_knn = 0

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train_scaled, y_train)
    predictions = knn.predict(x_test_scaled)
    accuracy = accuracy_score(predictions, y_test)

    if accuracy > best_score:
        best_score = accuracy
        best_k = k

best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(x_train_scaled, y_train)

print('Точность модели : ',best_score, "\n")
print("Лучший k = ", best_k, "\n")
