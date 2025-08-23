import pandas as pd

s=pd.Series([10,20,30,40])
print(s)

data = {"Name":['Alice','Bob','Sian','Dithin','Abin','Nehal','Sneha','Anju'],'Age':[25,30,21,25,26,20,24,20]}
df = pd.DataFrame(data)
print("###################")
print(df)
print("###################")
print(df.head())
print("###################")
print(df.tail())
print("###################")
print(df.info())
print("###################")
print(df.describe())
print("###################")
print(df.columns)
print("###################")
print(df.shape)