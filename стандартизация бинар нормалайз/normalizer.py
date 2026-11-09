# Normalizer() - работает ПО СТРОКАМ (по объектам), а не по столбцам.
# Приводит длину каждого вектора-строки к 1. Нужен для текстов (NLP) и поиска похожих объектов.
# (Для обычных таблиц вроде цен квартир или машин НЕ используется — там нужен StandardScaler).

from sklearn.preprocessing import Normalizer
import pandas as pd

# Пример: количество упоминаний слов в документах (разные тексты по длине)
data = {'слово_python': [10, 2, 0], 'слово_code': [5, 1, 0], 'слово_ai': [20, 4, 1]}
df = pd.DataFrame(data)

# Создаем нормализатор
normalizer = Normalizer()

# Normalizer масштабирует каждую строку независимо от других
df_normalized = normalizer.fit_transform(df)
print(df_normalized)


# x_train_scaled = normalizer.fit_transform(x_train)  # Обучает по строкам трена
# x_test_scaled  = normalizer.transform(x_test)       # Применяет те же правила к тесту