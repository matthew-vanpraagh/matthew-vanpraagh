import pandas as pd

df = pd.read_csv("/Users/matthewvanpraagh/Downloads/condensed.csv")
print(df.head())
df.iloc[:,1:]
df['Total'] = df.sum(axis=1)
head = df['Total'].value_counts()
print(head)
total = sum(head)
print(total)

total = df['Total']

df = df[df['Total'] != 2]

head = df['Total'].value_counts()
total = sum(head)
print(total)

df = df[df['Total'] != 3]

head = df['Total'].value_counts()
total = sum(head)
print(total)

df = df[df['Total'] != 4]

head = df['Total'].value_counts()
total = sum(head)
print(total)

df = df[df['Total'] != 5]

head = df['Total'].value_counts()
total = sum(head)
print(total)

df = df[df['Total'] != 6]

head = df['Total'].value_counts()
total = sum(head)
print(total)

df = df[df['Total'] != 7]

head = df['Total'].value_counts()
total = sum(head)
print(total)

df = df[df['Total'] != 8]

head = df['Total'].value_counts()
total = sum(head)
print(total)

df = df[df['Total'] != 9]

head = df['Total'].value_counts()
total = sum(head)
print(total)

df = df[df['Total'] != 10]

head = df['Total'].value_counts()
total = sum(head)
print(total)

df.to_csv("ones_removed.csv")