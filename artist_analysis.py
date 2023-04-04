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

data1 = "/Users/matthewvanpraagh/Downloads/MLHD_demographics.csv"
data2 = "/Users/matthewvanpraagh/Downloads/MLHD_behavioural_features.csv"

df_1 = pd.read_csv(data2, sep="\t", header=0)
columns = df_1.columns

print(df_1.shape)
input()

print(df_1.head())
print(columns)
print(len(columns))

mainstreamness = df_1["mainstreamness_artist"]

mean_exploratoryness = df_1["exploratoryness_artist"].mean()
mean_mainstreamness = df_1["mainstreamness_artist"].mean()
mean_genderedness = df_1["genderedness_artist"].mean()
mean_fringeness = df_1["fringeness_artist"].mean()
stdev_exploratoryness = df_1["exploratoryness_artist"].std()
stdev_mainstreamness = df_1["mainstreamness_artist"].std()
stdev_genderedness = df_1["genderedness_artist"].std()
stdev_fringeness = df_1["fringeness_artist"].std()
print(mainstreamness.head())
print(mean_exploratoryness)
print(mean_mainstreamness)
print(mean_genderedness)
print(mean_fringeness)
print(stdev_exploratoryness)
print(stdev_mainstreamness)
print(stdev_genderedness)
print(stdev_fringeness)

exploratoryness = df_1['exploratoryness_artist']
mainstreamness = df_1['mainstreamness_artist']
genderedness = df_1['genderedness_artist']
fringeness = df_1['fringeness_artist']

exploratoryness = (exploratoryness - mean_exploratoryness) / stdev_exploratoryness
mainstreamness = (mainstreamness - mean_mainstreamness) / stdev_mainstreamness
genderedness = (genderedness - mean_genderedness) / stdev_genderedness
fringeness = (fringeness - mean_fringeness) / stdev_fringeness

print(exploratoryness.head())
print(exploratoryness.describe())

mean_exploratoryness = exploratoryness.mean()
stdev_exploratoryness = exploratoryness.std()
mean_mainstreamness = mainstreamness.mean()
stdev_mainstreamness = mainstreamness.std()
mean_genderedness = genderedness.mean()
stdev_genderedness = genderedness.std()
mean_fringeness = fringeness.mean()
stdev_fringeness = fringeness.std()

"""
print("Normalized:")
print(mainstreamness.head())
print(mean_exploratoryness)
print(mean_mainstreamness)
print(mean_genderedness)
print(mean_fringeness)
print(stdev_exploratoryness)
print(stdev_mainstreamness)
print(stdev_genderedness)
print(stdev_fringeness)
"""

print(mean_exploratoryness)
print(mean_mainstreamness)
print(mean_genderedness)
print(mean_fringeness)
print(stdev_exploratoryness)
print(stdev_mainstreamness)
print(stdev_genderedness)
print(stdev_fringeness)
print(exploratoryness.max())
print(exploratoryness.min())
print(mainstreamness.max())
print(mainstreamness.min())
print(genderedness.max())
print(genderedness.min())
print(fringeness.max())
print(fringeness.min())

df_1['exploratoryness_artist'] = exploratoryness
df_1['mainstreamness_artist'] = mainstreamness
df_1['genderedness_artist'] = genderedness
df_1['fringeness_artist'] = fringeness
print(df_1['fringeness_artist'].mean())

#df_1.to_csv('normalized_data_1.csv')

sns.distplot(a=exploratoryness, bins=40)
plt.xlim(-5,5)
plt.title("Exploratoryness Normalized")
plt.show()
sns.distplot(a=mainstreamness, bins=40)
plt.xlim(-5,5)
plt.title("Mainstreamness Normalized")
plt.show()
sns.distplot(a=genderedness, bins=40)
plt.xlim(-5,5)
plt.title("Genderedness Normalized")
plt.show()
sns.distplot(a=fringeness, bins=40)
plt.xlim(-5,5)
plt.title("Fringeness Normalized")
plt.show()

# set a grey background (use sns.set_theme() if seaborn version 0.11.0 or above)
sns.set(style="darkgrid")

sns.histplot(data=df_1, x=exploratoryness, color="skyblue", label="Exploratoryness", kde=True)
sns.histplot(data=df_1, x=mainstreamness, color="red", label="Mainstreamness", kde=True)
sns.histplot(data=df_1, x=genderedness, color="green", label="Genderedness", kde=True)
sns.histplot(data=df_1, x=fringeness, color="yellow", label="Fringeness", kde=True)

plt.legend()
plt.xlim(-5,5)
plt.title("Overlay of Four Attributes")
plt.show()


exploratoryness = df_1['exploratoryness_artist'].sample(n=50000, random_state=1)
mainstreamness = df_1['mainstreamness_artist'].sample(n=50000, random_state=1)
genderedness = df_1['genderedness_artist'].sample(n=50000, random_state=1)
fringeness = df_1['fringeness_artist'].sample(n=50000, random_state=1)

print(exploratoryness.describe())
print(mainstreamness.describe())
print(genderedness.describe())
print(fringeness.describe())

plt.scatter(exploratoryness, mainstreamness)
plt.xlabel("Exploratoryness", fontsize=10)
plt.ylabel("Mainstreamness", fontsize=10)
plt.title("Exploratoryness vs. Mainstreamness")
plt.show()
r = np.corrcoef(exploratoryness, mainstreamness)
print(r)

plt.scatter(exploratoryness, fringeness)
plt.xlabel("Exploratoryness", fontsize=10)
plt.ylabel("Fringeness", fontsize=10)
plt.title("Exploratoryness vs. Fringeness")
plt.show()
r = np.corrcoef(exploratoryness, fringeness)
print(r)

plt.scatter(exploratoryness, genderedness)
plt.xlabel("Exploratoryness", fontsize=10)
plt.ylabel("Genderedness", fontsize=10)
plt.title("Exploratoryness vs. Genderedness")
plt.show()
r = np.corrcoef(exploratoryness, genderedness)
print(r)

plt.scatter(mainstreamness, fringeness)
plt.xlabel("Mainstreamness", fontsize=10)
plt.ylabel("Fringeness", fontsize=10)
plt.title("Mainstreamness vs. Fringeness")
plt.show()
r = np.corrcoef(mainstreamness, fringeness)
print(r)

plt.scatter(mainstreamness, genderedness)
plt.xlabel("Mainstreamness", fontsize=10)
plt.ylabel("Genderedness", fontsize=10)
plt.title("Mainstreamness vs. Genderedness")
plt.show()
r = np.corrcoef(mainstreamness, genderedness)
print(r)

plt.scatter(fringeness, genderedness)
plt.xlabel("Fringeness", fontsize=10)
plt.ylabel("Genderedness", fontsize=10)
plt.title("Fringeness vs. Genderedness")
plt.show()
r = np.corrcoef(fringeness, genderedness)
print(r)

plt.boxplot(exploratoryness)
plt.xlabel("Boxplot of Exploratoryness")
plt.show()
plt.boxplot(mainstreamness)
plt.xlabel("Boxplot of Mainstreamness")
plt.show()
plt.boxplot(fringeness)
plt.xlabel("Boxplot of Fringeness")
plt.show()
plt.boxplot(genderedness)
plt.xlabel("Boxplot of Genderedness")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
x = mainstreamness
y = exploratoryness
z = genderedness
ax.set_xlabel("Mainstreamness")
ax.set_ylabel("Exploratoryness")
ax.set_zlabel("Genderedness")
ax.scatter(x, y, z)
plt.show()

df_1_new = df_1.drop('uuid', axis=1)

from sklearn.decomposition import PCA
x = df_1_new.loc[:, ['exploratoryness_artist', 'mainstreamness_artist', 'genderedness_artist', 'fringeness_artist']].values
pca = PCA(n_components=2)
components = pca.fit_transform(x)
principalDf = pd.DataFrame(data=components, columns = ['principal component 1', 'principal component 2'])
print(principalDf.head())

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
ax.scatter(principalDf['principal component 1'], principalDf['principal component 2'])
colors = ['r', 'g']
ax.grid()
plt.show()

pc1 = principalDf['principal component 1']
pc2 = principalDf['principal component 2']

k = gaussian_kde(np.vstack([pc1, pc2]))
xi, yi = np.mgrid[pc1.min():pc1.max():pc1.size**0.5*1j,pc2.min():pc2.max():pc2.size**0.5*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))

fig = plt.figure(figsize=(7,8))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax1.set_title("Density Heatmap of PCA")
ax1.set_xlabel("Component 1")
ax1.set_ylabel("Component 2")
ax1.pcolormesh(xi, yi, zi.reshape(xi.shape), alpha=0.5)
ax2.contourf(xi, yi, zi.reshape(xi.shape), alpha=0.5)
ax1.set_xlim(-2, 2)
ax1.set_ylim(-2, 2)
ax2.set_xlim(-2, 2)
ax2.set_ylim(-2, 2)
plt.show()

df_1.to_csv('filtered_data.csv', index=True)