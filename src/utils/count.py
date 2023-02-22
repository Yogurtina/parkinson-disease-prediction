import pandas as pd
data = pd.read_csv('data.csv', delimiter = ',', header=0)

names = list(data['name'])

for i in range(1, 50 + 1):
    count = names.count(f"phon_R01_S{i:02d}")
    print(f"Number: {i} ", count)

