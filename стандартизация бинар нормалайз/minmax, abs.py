# 1. MinMaxScaler -> Зажимает все значения в строгий диапазон от 0 до 1.
# Идеально для нейросетей, картинок (пиксели от 0 до 255) и алгоритмов, которым нужны жесткие рамки.
from sklearn.preprocessing import MinMaxScaler

scaler_minmax = MinMaxScaler()
# x_train_scaled = scaler_minmax.fit_transform(x_train)
# x_test_scaled  = scaler_minmax.transform(x_test)
#Как работает: Находит минимальное и максимальное значение в колонке. Самое маленькое число становится равным 0, самое большое — 1, а все остальные превращаются в дроби между ними (например, 0.45, 0.82).


# 2. MaxAbsScaler -> Зажимает значения от -1 до 1, но ГЛАВНАЯ ФИШКА: оставляет нули нулями.
# Используется для «разреженных» данных (тексты, отзывы, матрицы оценок), где 99% — это нули,
# чтобы не забивать оперативную память компьютера случайными дробями.
from sklearn.preprocessing import MaxAbsScaler

scaler_maxabs = MaxAbsScaler()
# x_train_scaled = scaler_maxabs.fit_transform(x_train)
# x_test_scaled  = scaler_maxabs.transform(x_test)


# БОНУС: Любой скейлер можно повернуть назад (вернуть исходные числа, миллионы тенге и т.д.):
# original_data = scaler_maxabs.inverse_transform(x_train_scaled)