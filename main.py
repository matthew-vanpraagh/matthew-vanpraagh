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

print(df_1.head())
print(columns)
print(len(columns))

"""
mean_exploratoryness = df_1["exploratoryness_artist"].mean()
mean_mainstreamness = df_1["mainstreamness_artist"].mean()
mean_genderedness = df_1["genderedness_artist"].mean()
mean_fringeness = df_1["fringeness_artist"].mean()
stdev_exploratoryness = df_1["exploratoryness_artist"].std()
stdev_mainstreamness = df_1["mainstreamness_artist"].std()
stdev_genderedness = df_1["genderedness_artist"].std()
stdev_fringeness = df_1["fringeness_artist"].std()
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

exploratoryness = df_1['exploratoryness_artist'].sample(n=100000, random_state=1)
mainstreamness = df_1['mainstreamness_artist'].sample(n=100000, random_state=1)
genderedness = df_1['genderedness_artist'].sample(n=100000, random_state=1)
fringeness = df_1['fringeness_artist'].sample(n=100000, random_state=1)

print(exploratoryness.describe())
print(mainstreamness.describe())
print(genderedness.describe())
print(fringeness.describe())

plt.plot(exploratoryness, norm.pdf (exploratoryness, mean_exploratoryness, stdev_exploratoryness))
plt.title("Normalized plot for Exploratoryness")
plt.show()
plt.plot(mainstreamness, norm.pdf (mainstreamness, mean_mainstreamness, stdev_mainstreamness))
plt.title("Normalized plot for Mainstreamness")
plt.show()
plt.plot(genderedness, norm.pdf (genderedness, mean_genderedness, stdev_genderedness))
plt.title("Normalized plot for Genderedness")
plt.show()
plt.plot(fringeness, norm.pdf (fringeness, mean_fringeness, stdev_fringeness))
plt.title("Normalized plot for Fringeness")
plt.show()

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
"""

mainstreamness = df_1["mainstreamness_album"]

mean_exploratoryness = df_1["exploratoryness_album"].mean()
mean_mainstreamness = df_1["mainstreamness_album"].mean()
mean_genderedness = df_1["genderedness_album"].mean()
mean_fringeness = df_1["fringeness_album"].mean()
stdev_exploratoryness = df_1["exploratoryness_album"].std()
stdev_mainstreamness = df_1["mainstreamness_album"].std()
stdev_genderedness = df_1["genderedness_album"].std()
stdev_fringeness = df_1["fringeness_album"].std()
print(mainstreamness.head())
print(mean_exploratoryness)
print(mean_mainstreamness)
print(mean_genderedness)
print(mean_fringeness)
print(stdev_exploratoryness)
print(stdev_mainstreamness)
print(stdev_genderedness)
print(stdev_fringeness)

exploratoryness = df_1['exploratoryness_album']
mainstreamness = df_1['mainstreamness_album']
genderedness = df_1['genderedness_album']
fringeness = df_1['fringeness_album']

exploratoryness = (exploratoryness - mean_exploratoryness) / stdev_exploratoryness
mainstreamness = (mainstreamness - mean_mainstreamness) / stdev_mainstreamness
genderedness = (genderedness - mean_genderedness) / stdev_genderedness
fringeness = (fringeness - mean_fringeness) / stdev_fringeness

print(exploratoryness.head())
print(exploratoryness.describe())
input()

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



"""indexE = df_1[df_1["exploratoryness_album"] <= -5].index
df_1["exploratoryness_album"] = df_1["exploratoryness_album"].drop(indexE, inplace=True)
indexM = df_1[df_1["mainstreamness_album"] <= -3].index
df_1["mainstreamness_album"].drop(indexM, inplace=True)
indexG = df_1[df_1["genderedness_album"] <= -3].index
df_1["genderedness_album"].drop(indexG, inplace=True)
indexF = df_1[df_1["fringeness_album"] <= -3].index
df_1["fringeness_album"].drop(indexF, inplace=True)
index1 = df_1[df_1["exploratoryness_album"] >= 3].index
df_1.drop(indexE, inplace=True)
index2 = df_1[df_1["mainstreamness_album"] >= 3].index
df_1.drop(indexM, inplace=True)
index3 = df_1[df_1["genderedness_album"] >= 3].index
df_1.drop(indexG, inplace=True)
index4 = df_1[df_1["fringeness_album"] >= 3].index
df_1.drop(indexF, inplace=True)"""

def remove_outlier_IQR(df):
    Q1=df_1["exploratoryness_album".quantile(0.25)]
    Q3=df_1["exploratoryness_album".quantile(0.75)]
    IQR=Q3-Q1
    df_final=df[~((df<(Q1-1.5*IQR)) | (df>(Q3+1.5*IQR)))]
    return exploratoryness

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
input()

df_1['exploratoryness_album'] = exploratoryness
df_1['mainstreamness_album'] = mainstreamness
df_1['genderedness_album'] = genderedness
df_1['fringeness_album'] = fringeness
print(df_1['fringeness_album'].mean())
input()

df_1.to_csv('normalized_data_1.csv')

"""
exploratoryness = df_1['exploratoryness_album']
mainstreamness = df_1['mainstreamness_album']
genderedness = df_1['genderedness_album']
fringeness = df_1['fringeness_album']
"""

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


exploratoryness = df_1['exploratoryness_album'].sample(n=50000, random_state=1)
mainstreamness = df_1['mainstreamness_album'].sample(n=50000, random_state=1)
genderedness = df_1['genderedness_album'].sample(n=50000, random_state=1)
fringeness = df_1['fringeness_album'].sample(n=50000, random_state=1)

print(exploratoryness.describe())
print(mainstreamness.describe())
print(genderedness.describe())
print(fringeness.describe())

plt.plot(exploratoryness, norm.pdf (exploratoryness, mean_exploratoryness, stdev_exploratoryness))
plt.title("Normalized plot for Exploratoryness")
plt.show()
plt.plot(mainstreamness, norm.pdf (mainstreamness, mean_mainstreamness, stdev_mainstreamness))
plt.title("Normalized plot for Mainstreamness")
plt.show()
plt.plot(genderedness, norm.pdf (genderedness, mean_genderedness, stdev_genderedness))
plt.title("Normalized plot for Genderedness")
plt.show()
plt.plot(fringeness, norm.pdf (fringeness, mean_fringeness, stdev_fringeness))
plt.title("Normalized plot for Fringeness")
plt.show()

plt.scatter(exploratoryness, mainstreamness)
plt.xlabel("Exploratoryness", fontsize=10)
plt.ylabel("Mainstreamness", fontsize=10)
plt.title("Exploratoryness vs. Mainstreamness")
plt.show()
r = np.corrcoef(exploratoryness, mainstreamness)
print(r)

xy = np.vstack([exploratoryness,mainstreamness])
z = gaussian_kde(xy)(xy)
fig, ax = plt.subplots()
ax.scatter(exploratoryness, mainstreamness, c=z, s=100)
plt.xlabel("Exploratoryness", fontsize=10)
plt.ylabel("Mainstreamness", fontsize=10)
plt.title("Exploratoryness vs. Mainstreamness")
plt.show()

plt.scatter(exploratoryness, fringeness)
plt.xlabel("Exploratoryness", fontsize=10)
plt.ylabel("Fringeness", fontsize=10)
plt.title("Exploratoryness vs. Fringeness")
plt.show()
r = np.corrcoef(exploratoryness, fringeness)
print(r)

xy = np.vstack([exploratoryness,fringeness])
z = gaussian_kde(xy)(xy)
fig, ax = plt.subplots()
ax.scatter(exploratoryness, fringeness, c=z, s=100)
plt.xlabel("Exploratoryness", fontsize=10)
plt.ylabel("Fringeness", fontsize=10)
plt.title("Exploratoryness vs. Fringeness")
plt.show()

plt.scatter(exploratoryness, genderedness)
plt.xlabel("Exploratoryness", fontsize=10)
plt.ylabel("Genderedness", fontsize=10)
plt.title("Exploratoryness vs. Genderedness")
plt.show()
r = np.corrcoef(exploratoryness, genderedness)
print(r)

xy = np.vstack([exploratoryness,genderedness])
z = gaussian_kde(xy)(xy)
fig, ax = plt.subplots()
ax.scatter(exploratoryness, genderedness, c=z, s=100)
plt.xlabel("Exploratoryness", fontsize=10)
plt.ylabel("Genderedness", fontsize=10)
plt.title("Exploratoryness vs. Genderedness")
plt.show()

plt.scatter(mainstreamness, fringeness)
plt.xlabel("Mainstreamness", fontsize=10)
plt.ylabel("Fringeness", fontsize=10)
plt.title("Mainstreamness vs. Fringeness")
plt.show()
r = np.corrcoef(mainstreamness, fringeness)
print(r)

xy = np.vstack([mainstreamness, fringeness])
z = gaussian_kde(xy)(xy)
fig, ax = plt.subplots()
ax.scatter(mainstreamness, fringeness, c=z, s=100)
plt.xlabel("Mainstreamness", fontsize=10)
plt.ylabel("Fringeness", fontsize=10)
plt.title("Mainstreamness vs. Fringeness")
plt.show()

plt.scatter(mainstreamness, genderedness)
plt.xlabel("Mainstreamness", fontsize=10)
plt.ylabel("Genderedness", fontsize=10)
plt.title("Mainstreamness vs. Genderedness")
plt.show()
r = np.corrcoef(mainstreamness, genderedness)
print(r)

xy = np.vstack([mainstreamness, genderedness])
z = gaussian_kde(xy)(xy)
fig, ax = plt.subplots()
ax.scatter(mainstreamness, genderedness, c=z, s=100)
plt.xlabel("Mainstreamness", fontsize=10)
plt.ylabel("Genderedness", fontsize=10)
plt.title("Mainstreamness vs. Genderedness")
plt.show()

plt.scatter(fringeness, genderedness)
plt.xlabel("Fringeness", fontsize=10)
plt.ylabel("Genderedness", fontsize=10)
plt.title("Fringeness vs. Genderedness")
plt.show()
r = np.corrcoef(fringeness, genderedness)
print(r)

xy = np.vstack([fringeness, genderedness])
z = gaussian_kde(xy)(xy)
fig, ax = plt.subplots()
ax.scatter(fringeness, mainstreamness, c=z, s=100)
plt.xlabel("Fringeness", fontsize=10)
plt.ylabel("Genderedness", fontsize=10)
plt.title("Fringeness vs. Genderedness")
plt.show()

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

df_1.to_csv('filtered_data.csv', index=True)