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
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import gaussian_kde
from sklearn.metrics import jaccard_score

df = pd.read_csv("ones_removed.csv")

#df_condensed = df.iloc[0:25000]

subset = df[['Artist', 'Track', 'Album']]
tuples = [tuple(x) for x in subset.to_numpy()]
df['new_tuples'] = list(zip(df['Artist'], df['Track'], df['Album']))

print(df['new_tuples'].head())

keep = df['new_tuples']

from sklearn.metrics.pairwise import pairwise_distances
jaccard = 1 - pairwise_distances(df.to_numpy(), metric='jaccard')

df = pd.DataFrame(jaccard)

print(df.columns)

df['Records'] = keep

df.set_index('Records')

print(df.shape)

df.to_csv('jaccard_sample_2.csv')
print('Csv printed')

print(jaccard)


"""
song1 = input("Name of first song: ")
song2= input("Name of second song: ")
artist1 = input("Name of first artist: ")
artist2 = input("Name of second artist: ")
"""



#jaccard_score()
