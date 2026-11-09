import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# 1. Создаем исходный датасет с текстовыми колонками (категориями)
data = {
    'Car_ID': [1, 2, 3, 4, 5],
    'Brand': ['Toyota', 'BMW', 'Toyota', 'Audi', 'BMW'],
    'Color': ['Red', 'Blue', 'Blue', 'Red', 'Black'],
    'Price': [15000, 30000, 16000, 25000, 32000],
}
df = pd.DataFrame(data)

cat_columns = ['Brand', 'Color']

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
# - sparse_output=False: возвращает обычную матрицу (таблицу), а не сжатый объект
# - handle_unknown='ignore': если в будущем попадется новый бренд, код не упадет с ошибкой

encoded_array = encoder.fit_transform(df[cat_columns])

encoded_df_names = encoder.get_feature_names_out(cat_columns)

df_encoded = pd.DataFrame(encoded_array, columns=encoded_df_names)

df_final = pd.concat([df.drop(cat_columns, axis=1), df_encoded], axis=1)

print(df_final)


# это паттерн, изменить нужно будет только название колонок в cat, и их количество, если надо