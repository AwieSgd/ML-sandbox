from sklearn.preprocessing import Binarizer
import numpy as np

a = np.array([[5, 2, 4, 3, 0]])

# 1. Создаем объект трансформера
biner = Binarizer(threshold=3)

# 2. Применяем transform к данным
binarized = biner.transform(a)

print(binarized)
# Выведет: [[1 0 1 0 0]]  (потому что (5 и 4) > 3, а остальные <= 3)