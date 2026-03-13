import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

df = pd.read_csv("sales_data.csv")

reg_df = df.drop(["Date","Day","Month","Customer_Age","Age_Group",
"Customer_Gender","Country","State",
"Product_Category","Sub_Category","Product"], axis=1)

x = reg_df.drop("Revenue", axis=1)
y = reg_df["Revenue"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(x_train, y_train)

y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("Train R2:", train_r2*100)
print("Test R2 :", test_r2*100)
print("MAE:", mean_absolute_error(y_test, y_test_pred))
print("MSE:", mean_squared_error(y_test, y_test_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_test_pred)))

if train_r2 - test_r2 > 0.15:
    print("Model is Overfitting")
elif train_r2 < 0.50 and test_r2 < 0.50:
    print("Model is Underfitting")
else:
    print("Model is Balanced")

cv_scores = cross_val_score(model, x, y, cv=5, scoring="r2")
print("Cross Validation R2 Mean:", np.mean(cv_scores)*100)

importance = pd.Series(model.feature_importances_, index=x.columns).sort_values(ascending=False)
print("\nTop Features:")
print(importance)

plt.figure()
plt.scatter(y_test, y_test_pred)
plt.xlabel("Actual Revenue")
plt.ylabel("Predicted Revenue")
plt.title("Actual vs Predicted Revenue")
plt.show()

residuals = y_test - y_test_pred
plt.figure()
plt.scatter(y_test_pred, residuals)
plt.axhline(y=0)
plt.xlabel("Predicted Revenue")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

next_year = df["Year"].max() + 1
future_data = x.iloc[-1:].copy()
future_data["Year"] = next_year
predicted_next_year = model.predict(future_data)
print("\nNext Year:", next_year)
print("Predicted Revenue:", predicted_next_year[0])


customer_data = df.groupby("Customer_Age").agg({
    "Revenue":"sum",
    "Profit":"sum",
    "Order_Quantity":"sum"
}).reset_index()

X_seg = customer_data[["Revenue","Profit","Order_Quantity","Customer_Age"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_seg)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1,11), wcss)
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method for Customer Segmentation")
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
customer_data["Cluster"] = kmeans.fit_predict(X_scaled)

score = silhouette_score(X_scaled, customer_data["Cluster"])
print("Silhouette Score:", score)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
customer_data["PCA1"] = X_pca[:,0]
customer_data["PCA2"] = X_pca[:,1]

plt.figure()
for cluster in customer_data["Cluster"].unique():
    subset = customer_data[customer_data["Cluster"] == cluster]
    plt.scatter(subset["PCA1"], subset["PCA2"], label=f"Cluster {cluster}")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Customer Segmentation Clusters")
plt.legend()
plt.show()

plt.figure()
customer_data.groupby("Cluster")["Revenue"].mean().plot(kind="bar")
plt.title("Average Revenue per Cluster")
plt.ylabel("Revenue")
plt.show()

plt.figure()
customer_data.groupby("Cluster")["Profit"].mean().plot(kind="bar")
plt.title("Average Profit per Cluster")
plt.ylabel("Profit")
plt.show()