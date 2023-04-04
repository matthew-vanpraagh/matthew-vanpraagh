import math
import numpy as np
import seaborn as sns
import datetime
from scipy import stats
import plotly.express as px
import pandas as pd
import plotly.graph_objs as go
import warnings
warnings.filterwarnings("ignore")
import kmeans1d
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import gaussian_kde

df = pd.read_csv("combined_listening.csv")

print(df.shape())

print(df.head())

df.iloc[:,4:]
df['Total'] = df.sum(axis=1)
df['Total'].value_counts()

input()



print(df.columns)
for df.iloc[:, [4, 103]] in df:
    for df.iloc[[1, 10000]] in df:
        df[df != 0] = 1

smaller_df = df.iloc[0:25000]

merge = smaller_df.pivot_table(index=['Artist', 'Album', 'Track'], aggfunc='count')

print(merge.columns)
print(merge.head())
print(len(merge.index))

merge.to_csv("condensed.csv")
print("Csv printed")

