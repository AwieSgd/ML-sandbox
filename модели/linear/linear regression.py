import ssl
from sklearn.linear_model import LinearRegression

ssl._create_default_https_context = ssl._create_unverified_context
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing


data = fetch_california_housing(as_frame=True)
df = data.frame

X = df.drop(columns=['MedHouseVal'])
y = df['MedHouseVal']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
predictions = model.predict(x_test)
print(predictions[:5])
print(score)



