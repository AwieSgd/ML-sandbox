import ssl


ssl._create_default_https_context = ssl._create_unverified_context
from sklearn.model_selection import train_test_split

import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix



df = sns.load_dataset("titanic")

df['sex'] = df['sex'].map({'male': 0, 'female': 1})

df['age'] = df['age'].fillna(df['age'].median())
df['fare'] = df['fare'].fillna(df['fare'].median())


x = df[[ "sex", "age", "fare", "parch"]]
y = df['survived']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)



model = RandomForestClassifier(n_estimators=100 , random_state=42)
model.fit(x_train, y_train)

predictions = model.predict(x_test)
accuracy = accuracy_score(predictions,y_test)
print(accuracy)

confusion_matrix = confusion_matrix(y_test,predictions)
print(confusion_matrix)
print(predictions[:10], '\n')
print(y_test[:10], '\n')


#[86 19] - 86 умерших угадал, но 18 умерших посчитал как выжившими
#[18 56] - 56 выживших угадал, но 18 выживших посчитал как умершими



