from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
iris = load_iris()
print(iris.feature_names, "\n")
print(iris.target_names)
print(iris.data.shape)

X = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train_scaled, y_train)

predictions = model.predict(x_test_scaled)
probability = model.predict_proba(x_test_scaled)
accuracy = accuracy_score(predictions, y_test)
print(predictions[:10], '\n')
print(probability[:10], '\n')
print(y_test[:10], '\n')
print(accuracy)
