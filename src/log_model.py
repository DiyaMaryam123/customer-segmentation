import mlflow
import mlflow.sklearn
import joblib
import os

# Configure MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")

mlflow.set_experiment("Customer_Segmentation")

# Load existing trained model using joblib
model = joblib.load("models/kmeans_model.pkl")

# Start MLflow run
with mlflow.start_run(run_name="KMeans_Model_v1"):

    # Log model parameters
    mlflow.log_param("algorithm", "KMeans")
    mlflow.log_param("n_clusters", 5)
    mlflow.log_param("random_state", 42)

    # Log evaluation metrics
    mlflow.log_metric("silhouette_score", 0.5547)
    mlflow.log_metric("davies_bouldin_score", 0.5722)

    # Log actual trained model
    mlflow.sklearn.log_model(
        sk_model=model,
        name="kmeans_model"
    )

    # Log scaler
    mlflow.log_artifact(
        "models/scaler.pkl",
        artifact_path="preprocessing"
    )

    # Log metadata
    mlflow.log_artifact(
        "models/model_metadata.json"
    )

    print("Model and artifacts logged successfully!")
