#StandartScaler() - нужен для того, если у признаков огромная разница, KNN, SVM - Must have

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

data = {
    'Комнаты': [1, 2, 3, 4, 5],
    'Площадь_кв_м': [38, 55, 72, 105, 140],
    'Цена_тенге': [18000000, 26000000, 42000000, 68000000, 115000000],
}
df = pd.DataFrame(data)
x = df[['Комнаты', 'Площадь_кв_м']]
y = df['Цена_тенге']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)

# 3. Создаем скейлер и масштабируем X
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
# Обучаем скейлер на трене и меняем его
x_test_scaled = scaler.transform( x_test)
# Тест только трансформируем (без fit!)

model = LinearRegression()
model.fit(x_train_scaled, y_train)
score = model.score(x_test_scaled, y_test)
print(score)

#x_train - всегда идет с fit_transform
#x_test - идет без fit, и выходит transform(x_test)

#######################################################################################################################
# Исходный датасет по машинам
data1 = {
    'Объем_двигателя_л': [1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
    'Пробег_км': [150000, 120000, 80000, 50000, 30000, 10000],
    'Цена_тенге': [4500000, 6000000, 8500000, 12000000, 16000000, 21000000],
}

df1 = pd.DataFrame(data1)

x1 = df1[['Объем_двигателя_л', 'Пробег_км']]
y1 = df1['Цена_тенге']
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.4, random_state=42)

scaler1 = StandardScaler()
x1_train_scaled = scaler1.fit_transform(x1_train)
x1_test_scaled = scaler1.transform(x1_test)

model1 = LinearRegression()
model1.fit(x1_train_scaled, y1_train)
score1 = model1.score(x1_test_scaled, y1_test)
print(score1)
