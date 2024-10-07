#k-means Algoritham(clustring Algoritham )

#cluster-it is called as a group derived from the given data

#PROBLEM STAEMENT :-Find the targate customers for an mall based on their 
#salary and the time spend on the mall and grouped them into cluster
#here there are three cluster 0,1,2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#genrate dataset
np.random.seed(42)#function of numpy for reuseblity
data={
    'annual_income':np.random.randint(30000,100000,100),
    'spending-score':np.random.randint(1,100,100),
    
}
#(starting_annual_income,max_income,number_of_genrated_values)

df=pd.DataFrame(data)
print(df)
#plt.scatter(df['annual_income'],df['spending-score'])
#plt.show()
#apply the k_means algoritham on data
x=df.values
kmeans=KMeans(n_clusters=3,random_state=42)
df['cluster']=kmeans.fit_predict(x)

print(df)

plt.scatter(df['annual_income'],df['spending-score'],c=df['cluster'],cmap='rainbow')
plt.title("k-means clustring ")
plt.xlabel("annual salary")
plt.ylabel("spending score")
plt.show()

user_input={'annual_income':[900],'spending-score':[76]}
user_df=pd.DataFrame(user_input)
user_cluster=kmeans.predict(user_df)
print(f"the user belongs to cluster:{user_cluster[0]}")

if  user_cluster[0]==0:
    print("the customers are very good approchable")
elif user_cluster[0]==1:
    print("the users are avrage in approch")  
else:
    print("the users are poor audiance")      

