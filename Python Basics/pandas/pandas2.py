import pandas as pd

data = {
    "Name":['Alice','Bob','Sian','Dithin','Abin','Nehal','Sneha','Anju'],
    'Age':[25,30,21,25,None,20,24,20],
    'City':['Delhi','Mumbai','Chennai','Calicut','Banglore','Mysore','Kochi','Kannur']
    }

df = pd.DataFrame(data,index=[1,2,3,4,5,6,7,8])
print(df.loc[1])
print("######################")
print(df.loc[2,'Name'])
print("######################")
print(df.loc[:,['Name','City']])


print("######################")
print(df.iloc[0])
print("######################")
print(df.iloc[1,0])
print("######################")
print(df.iloc[:,0:2])
print("######################")
print(df.isnull())
print("######################")
print(df.dropna())
print("######################")
# print(df.fillna(28))
print("######################")
df['Age'].fillna(df['Age'].mean(),inplace=True)
print(df)