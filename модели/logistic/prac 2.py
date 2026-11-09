from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df = df[df['target'] != 2]

x = df.drop('target', axis=1)
y = df['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
scaled_train = scaler.fit_transform(x_train)
scaled_test = scaler.transform(x_test)

model = LogisticRegression()
model.fit(scaled_train, y_train)

predictions = model.predict(scaled_test)
probability = model.predict_proba(scaled_test)
accuracy = accuracy_score(y_test, predictions)

print(accuracy)
print(predictions[:10])
print(probability[:10])
print(y_test[:10])

