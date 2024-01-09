import pandas as pd

df = pd.read_csv("rewards.csv")
print(df.loc[:10])