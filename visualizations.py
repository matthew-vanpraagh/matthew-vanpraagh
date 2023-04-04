import math
import numpy as np
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import kmeans1d
import matplotlib.pyplot as plt

data1 = "/Users/matthewvanpraagh/PycharmProjects/MusicViz/normalized_data_1.csv"

df_1 = pd.read_csv(data1, sep=",", header=0)
columns = df_1.columns

print(df_1.head())
print(columns)
print(len(columns))

df_1 = df_1.sample(n = 10000, axis='rows')

exploratoryness = df_1['exploratoryness_album']
mainstreamness = df_1['mainstreamness_album']
genderedness = df_1['genderedness_album']
fringeness = df_1['fringeness_album']

df_1_new = df_1.drop('uuid', axis=1)

from scipy.stats.kde import gaussian_kde
k = gaussian_kde(np.vstack([mainstreamness, genderedness]))
xi, yi = np.mgrid[mainstreamness.min():mainstreamness.max():mainstreamness.size**0.5*1j,genderedness.min():genderedness.max():genderedness.size**0.5*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))

fig = plt.figure(figsize=(7,8))
ax1 = fig.add_subplot(212)
#ax2 = fig.add_subplot(212)
ax1.set_title('Density Heatmap of Mainstreamness vs. Genderedness')
ax1.set_xlabel("Mainstreamness")
ax1.set_ylabel("Genderedness")
ax1.contourf(xi, yi, zi.reshape(xi.shape), alpha=0.5)
#ax2.contourf(xi, yi, zi.reshape(xi.shape), alpha=0.5)
ax1.set_xlim(-2, 2)
ax1.set_ylim(-2, 2)
#ax2.set_xlim(-2, 2)
#ax2.set_ylim(-2, 2)
plt.show()

k = gaussian_kde(np.vstack([fringeness, genderedness]))
xi, yi = np.mgrid[fringeness.min():fringeness.max():fringeness.size**0.5*1j,genderedness.min():genderedness.max():genderedness.size**0.5*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))

fig = plt.figure(figsize=(7,8))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax1.set_title("Density Heatmap of Fringeness vs. Genderedness")
ax1.set_xlabel("Fringeness")
ax1.set_ylabel("Genderedness")
ax1.pcolormesh(xi, yi, zi.reshape(xi.shape), alpha=0.5)
ax2.contourf(xi, yi, zi.reshape(xi.shape), alpha=0.5)
ax1.set_xlim(-2, 2)
ax1.set_ylim(-2, 2)
ax2.set_xlim(-2, 2)
ax2.set_ylim(-2, 2)
plt.show()

from sklearn.decomposition import PCA
x = df_1_new.loc[:, ['exploratoryness_album', 'mainstreamness_album', 'genderedness_album', 'fringeness_album']].values
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

fig = plt.figure(figsize=(7,4))
ax1 = fig.add_subplot(212)
#ax2 = fig.add_subplot(212)
ax1.set_title("Density Heatmap of PCA")
ax1.set_xlabel("Component 1")
ax1.set_ylabel("Component 2")
ax1.contourf(xi, yi, zi.reshape(xi.shape), alpha=0.5)
#ax2.contourf(xi, yi, zi.reshape(xi.shape), alpha=0.5)
ax1.set_xlim(-2, 2)
ax1.set_ylim(-2, 2)
#ax2.set_xlim(-2, 2)
#ax2.set_ylim(-2, 2)
plt.show()

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

print(exploratoryness.describe())
print(mainstreamness.describe())
print(genderedness.describe())
print(fringeness.describe())
""""""
plt.scatter(exploratoryness, mainstreamness)
plt.xlabel("Exploratoryness", fontsize=10)
plt.ylabel("Mainstreamness", fontsize=10)
plt.title("Exploratoryness vs. Mainstreamness")
plt.show()
r = np.corrcoef(exploratoryness, mainstreamness)[0,1]
print(r)
"""
xy = np.vstack([exploratoryness,mainstreamness])
z = gaussian_kde(xy)(xy)
fig, ax = plt.subplots()
ax.scatter(exploratoryness, mainstreamness, c=z, s=100)
plt.xlabel("Exploratoryness", fontsize=10)
plt.ylabel("Mainstreamness", fontsize=10)
plt.title("Exploratoryness vs. Mainstreamness")
plt.show()
"""
plt.scatter(exploratoryness, fringeness)
plt.xlabel("Exploratoryness", fontsize=10)
plt.ylabel("Fringeness", fontsize=10)
plt.title("Exploratoryness vs. Fringeness")
plt.show()
r = np.corrcoef(exploratoryness, fringeness)[0,1]
print(r)
"""
xy = np.vstack([exploratoryness,fringeness])
z = gaussian_kde(xy)(xy)
fig, ax = plt.subplots()
ax.scatter(exploratoryness, fringeness, c=z, s=100)
plt.xlabel("Exploratoryness", fontsize=10)
plt.ylabel("Fringeness", fontsize=10)
plt.title("Exploratoryness vs. Fringeness")
plt.show()
"""
plt.scatter(exploratoryness, genderedness)
plt.xlabel("Exploratoryness", fontsize=10)
plt.ylabel("Genderedness", fontsize=10)
plt.title("Exploratoryness vs. Genderedness")
plt.show()
r = np.corrcoef(exploratoryness, genderedness)[0,1]
print(r)
"""
xy = np.vstack([exploratoryness,genderedness])
z = gaussian_kde(xy)(xy)
fig, ax = plt.subplots()
ax.scatter(exploratoryness, genderedness, c=z, s=100)
plt.xlabel("Exploratoryness", fontsize=10)
plt.ylabel("Genderedness", fontsize=10)
plt.title("Exploratoryness vs. Genderedness")
plt.show()
"""
plt.scatter(mainstreamness, fringeness)
plt.xlabel("Mainstreamness", fontsize=10)
plt.ylabel("Fringeness", fontsize=10)
plt.title("Mainstreamness vs. Fringeness")
plt.show()
r = np.corrcoef(mainstreamness, fringeness)[0,1]
print(r)
"""
xy = np.vstack([mainstreamness, fringeness])
z = gaussian_kde(xy)(xy)
fig, ax = plt.subplots()
ax.scatter(mainstreamness, fringeness, c=z, s=100)
plt.xlabel("Mainstreamness", fontsize=10)
plt.ylabel("Fringeness", fontsize=10)
plt.title("Mainstreamness vs. Fringeness")
plt.show()
"""
plt.scatter(mainstreamness, genderedness)
plt.xlabel("Mainstreamness", fontsize=10)
plt.ylabel("Genderedness", fontsize=10)
plt.title("Mainstreamness vs. Genderedness")
plt.show()
r = np.corrcoef(mainstreamness, genderedness)[0,1]
print(r)
"""
xy = np.vstack([mainstreamness, genderedness])
z = gaussian_kde(xy)(xy)
fig, ax = plt.subplots()
ax.scatter(mainstreamness, genderedness, c=z, s=100)
plt.xlabel("Mainstreamness", fontsize=10)
plt.ylabel("Genderedness", fontsize=10)
plt.title("Mainstreamness vs. Genderedness")
plt.show()
"""
plt.scatter(fringeness, genderedness)
plt.xlabel("Fringeness", fontsize=10)
plt.ylabel("Genderedness", fontsize=10)
plt.title("Fringeness vs. Genderedness")
plt.show()
r = np.corrcoef(fringeness, genderedness)[0,1]
print(r)
"""
xy = np.vstack([fringeness, genderedness])
z = gaussian_kde(xy)(xy)
fig, ax = plt.subplots()
ax.scatter(fringeness, mainstreamness, c=z, s=100)
plt.xlabel("Fringeness", fontsize=10)
plt.ylabel("Genderedness", fontsize=10)
plt.title("Fringeness vs. Genderedness")
plt.show()
"""

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