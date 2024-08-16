#Gerekli kütüphaneleri içe aktaralım.
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt


#Veri setini datframe'e çevirelim.
df = pd.read_csv("Avm_Musterileri.csv")
#print(df.head())


#Veri setimize bir bakalım.
#plt.scatter(df["Annual Income (k$)"], df["Spending Score (1-100)"])
#plt.xlabel("Annual Income (k$)")
#plt.ylabel("Spending Score (1-100)")
#plt.show()


#Veri setinde bulunan başlıklar çok uzun. Bu nedenle ismlerini değiştirelim.
df.rename(columns={"Annual Income (k$)":"income"}, inplace=True)
df.rename(columns={"Spending Score (1-100)":"score"}, inplace=True)


#Şimdi verilerimizi normalize edelim.
#Bu  işlem için sklearn kütüphanesinde bulunan MinMaxScaler() fonksiyonunu kullanacağız.
scaler =MinMaxScaler()

scaler.fit(df[["income"]])
df["income"] = scaler.transform(df[["income"]])

scaler.fit(df[["score"]])
df["score"] = scaler.transform(df[["score"]])

#print(df.head())


#Şimdi Elbow yöntemini kullanarak K değerini belirleyelim.
k_range = range(1, 11)
list_dist = []
#for k in k_range:
#    kmeans = KMeans(n_clusters=k)
#    kmeans.fit(df[["income", "score"]])
#    list_dist.append(kmeans.inertia_)

#plt.xlabel("K")
#plt.ylabel("Distortion Değeri(Inertia)")
#plt.plot(k_range, list_dist)
#plt.savefig("sonuc1.png", dpi=300)
#plt.show()
#Görüldüğü üzeri K değeri 5 olarak bulunmuştur.Bundan sonra algoritmada k=5 olarak kullanılacaktır.


kmeans= KMeans(n_clusters=5)
y_predicted = kmeans.fit_predict(df[["income", "score"]])
#print(y_predicted)


df["cluster"] = y_predicted
#print(df.head())


df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]
df4 = df[df.cluster == 3]
df5 = df[df.cluster == 4]

plt.xlabel("income")
plt.ylabel("score")
plt.scatter(df1["income"], df1["score"], color="green")
plt.scatter(df2["income"], df2["score"], color="red")
plt.scatter(df3["income"], df3["score"], color="black")
plt.scatter(df4["income"], df4["score"], color="orange")
plt.scatter(df5["income"], df5["score"], color="purple")

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], color="blue", marker="X", label="Centroid")
plt.legend()
plt.savefig("sonuc2.png", dpi=300)
plt.show()