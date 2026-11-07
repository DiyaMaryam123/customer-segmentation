import mlflow
import mlflow.sklearn

# Configure MLflow tracking
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Create or select experiment
mlflow.set_experiment("Customer_Segmentation")

with mlflow.start_run(run_name="KMeans_Clustering_v1"):

    # Log model parameters
    mlflow.log_param("algorithm", "KMeans")
    mlflow.log_param("n_clusters", 5)
    mlflow.log_param("features", "Income, SpendingScore")
    mlflow.log_param("random_state", 42)

    # Log evaluation metrics
    mlflow.log_metric("silhouette_score", 0.5547)
    mlflow.log_metric("davies_bouldin_score", 0.5722)

    print("MLflow experiment logged successfully!")
