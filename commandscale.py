import matplotlib.pyplot as plt
import numpy as np
from sklearn import manifold
import pandas as pd
import sklearn as sk

read_data = pd.read_csv("jaccard_sample_2.csv")
df = pd.DataFrame(read_data)
labels = pd.read_csv(r"C:\Users\matthewvanpraagh\Downloads\condensed.csv")

print(labels)

print(labels.shape)

print(labels.columns)
labels['new_tuples'] = list(zip(labels['Artist'], labels['Track'], labels['Album']))

df.to_numpy()

print(df.shape)
"""
pca_songs = PCA(n_components=2)
principalComponents_songs = pca_songs.fit_transform(df)
df = principalComponents_songs
print(df.shape)
"""
#df = pd.DataFrame(df, columns = ['PCA 1', 'PCA 2'])

print(df.head)

#songs = labels["Track"]
d = np.array([df])

d = np.delete(d, 24390, 2)

print(d)

print(d.shape)

d = d.reshape(d.shape[1], d.shape[2])

print(d)

print(d.shape)

d = np.delete(d, 0, 1)

print(d)

print(d.shape)

d = np.array(d, dtype = float)

d = sk.utils.validation.check_symmetric(d, tol=1e-10, raise_warning=True, raise_exception=False)

mds = manifold.MDS(1001, dissimilarity='precomputed')
coords = mds.fit_transform(d)
x, y = coords[:, 0], coords[:, 1]

fig, ax = plt.subplots()
ax.scatter(x, y)
for (song, _x, _y) in zip(songs, x, y):
    ax.annotate(song, (_x, _y))
plt.show()
