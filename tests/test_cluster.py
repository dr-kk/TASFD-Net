from src.clustering.cluster_features import FeatureCluster

cluster=FeatureCluster()

result=cluster.cluster_folder(
    "features"
)

print(result)