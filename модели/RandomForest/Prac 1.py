import ssl
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
ssl._create_default_https_context = ssl._create_unverified_context

df = sns.load_dataset("penguins")
df.dropna(subset = ['sex'], inplace = True)

df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})

df['island'] = df['island'].map({'Torgersen': 0, 'Biscoe': 1, 'Dream': 2})

df['species'] = df['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2})


x = df.drop(['species'], axis = 1)
y = df['species']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
model = RandomForestClassifier(n_estimators = 100)
model.fit(x_train, y_train)

predictions = model.predict(x_test)
probability = model.predict_proba(x_test)
classification_report = classification_report(y_test, predictions)

print(classification_report)
