import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import seaborn as sns

df1 = pd.read_csv("rewards_no_credit.csv")
df2 = pd.read_csv("rewards_cedric.csv")
df3 = pd.read_csv("rewards_counterfactual.csv")
df4 = pd.read_csv("rewards_shared.csv")
#print(df1, df2)

sns.set_style("whitegrid")
agent = 5
df = pd.DataFrame({'no credit': df1[str(agent)], 'cedric': df2[str(agent)], 'counterfactual': df3[str(agent)], "shared": df4[str(agent)]})
for i in range(1, len(df)):
    df.iloc[i] = df.iloc[i] + df.iloc[i-1]
df.plot(xlabel='training episodes', ylabel='cumulative reward')
plt.savefig('reward.png')