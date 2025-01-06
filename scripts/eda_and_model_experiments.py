# eda_and_model_experiments.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load datasets
def load_data(train_path="data/Train.csv", test_path="data/Test.csv"):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data

# Perform EDA
def perform_eda(train_data):
    print("\n--- Performing EDA ---\n")
    
    # Target variable distribution
    print("Target Variable Distribution:")
    print(train_data["Segmentation"].value_counts())
    sns.countplot(data=train_data, x="Segmentation", palette="viridis")
    plt.title("Target Variable Distribution")
    plt.show()

    # Correlation heatmap for numerical features
    numeric_features = ["Age", "Work_Experience", "Family_Size"]
    print("\nCorrelation Matrix:")
    print(train_data[numeric_features].corr())
    sns.heatmap(train_data[numeric_features].corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()

    # Feature distributions
    train_data[numeric_features].hist(bins=20, figsize=(15, 10), color="skyblue", edgecolor="black")
    plt.suptitle("Feature Distributions")
    plt.show()

# Prototype clustering (KMeans)
def prototype_kmeans(train_data):
    print("\n--- Prototyping KMeans Clustering ---\n")
    
    # Select numerical features for clustering
    numeric_features = ["Age", "Work_Experience", "Family_Size"]
    clustering_data = train_data[numeric_features]

    # Impute missing values
    imputer = SimpleImputer(strategy="mean")
    clustering_data = imputer.fit_transform(clustering_data)

    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(clustering_data)
    
    # Fit KMeans with 4 clusters (matching the number of segments)
    kmeans = KMeans(n_clusters=4, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_features)
    
    # Evaluate clustering
    silhouette_avg = silhouette_score(scaled_features, cluster_labels)
    print(f"Silhouette Score: {silhouette_avg:.4f}")
    
    # Add cluster labels to the dataset for visualization
    train_data["Cluster"] = cluster_labels
    sns.countplot(data=train_data, x="Cluster", palette="viridis")
    plt.title("Cluster Distribution (KMeans)")
    plt.show()

# Main script
if __name__ == "__main__":
    # Load datasets
    train_data, test_data = load_data()

    # Perform EDA
    perform_eda(train_data)

    # Prototype KMeans clustering
    prototype_kmeans(train_data)

    print("\n--- Script Complete ---")