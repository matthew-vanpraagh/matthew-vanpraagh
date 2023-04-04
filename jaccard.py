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
from sklearn.metrics import jaccard_score

df = pd.read_csv("ones_removed.csv")

#df_condensed = df.iloc[0:25000]

from sklearn.metrics.pairwise import pairwise_distances
jaccard = 1 - pairwise_distances(df.to_numpy(), metric='jaccard')

print(jaccard)

pd.DataFrame(jaccard).to_csv('jaccard_sample_2.csv')
#jaccard.to_csv("jaccard_sample.csv")
print("Csv printed")

"""
song1 = input("Name of first song: ")
song2= input("Name of second song: ")
artist1 = input("Name of first artist: ")
artist2 = input("Name of second artist: ")
"""



#jaccard_score()
