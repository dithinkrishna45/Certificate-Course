import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns

# 1.load Dataset
data = pd.read_csv("Mall_Customers/Mall_Customers.csv",encoding='latin1')
print(data.head())

x = data.iloc[:,[3,4]].values
print(x)

wcss=[]
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init = "k-means++",random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

# 4. Plot elbow curve
sns.set()
plt.plot(range(1,11),wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

Cluster = 5

kmeans=KMeans(n_clusters=5,init='k-means++',random_state=0)

Y=kmeans.fit_predict(x)
print(Y)
plt.figure(figsize=(8,8))
plt.scatter(x[Y==0,0],x[Y==0,1],s=50,c='blue',label='Cluster 1')
plt.scatter(x[Y==1,0],x[Y==1,1],s=50,c='green',label='Cluster 2')
plt.scatter(x[Y==2,0],x[Y==2,1],s=50,c='pink',label='Cluster 3')
plt.scatter(x[Y==3,0],x[Y==3,1],s=50,c='black',label='Cluster 4')
plt.scatter(x[Y==4,0],x[Y==4,1],s=50,c='gray',label='Cluster 5')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='yellow',label='Centroids')
plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()