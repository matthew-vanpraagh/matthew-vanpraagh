import pandas as pd
import warnings
warnings.filterwarnings("ignore")

"""a = pd.read_csv("/Users/matthewvanpraagh/Documents/last.fm_anonymous/last1.csv", sep=';', header=0)
df = pd.DataFrame(a)

print(df.columns)

input("Hello")
"""

userData = pd.DataFrame()

i = 1
for i in range(1, 101):
    i = str(i)
    read_data = pd.read_csv("/Users/matthewvanpraagh/Documents/last.fm_anonymous/last" + i + ".csv", sep=';', header=0)
    temp_df = pd.DataFrame(read_data)
    merge = pd.concat([userData, temp_df])
    userData = merge
    print("Just added dataset " + str(i))
    #userData.append(temp_df)
    i = int(i)
    i = i + 1
    #print(temp_df.columns)

print(merge.columns)
print(merge.head())
print(len(merge.index))

merge.to_csv("combined_listening.csv")

#print(userData.head())

#print(userData.columns)