import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")

client = MlflowClient()

experiment = client.get_experiment_by_name(
    "Customer_Segmentation"
)

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["attributes.start_time DESC"],
    max_results=10
)

run = next(
    r for r in runs
    if r.data.tags.get("mlflow.runName") == "KMeans_Model_v1"
)

model_uri = f"runs:/{run.info.run_id}/kmeans_model"

model_name = "Customer_Segmentation_KMeans"

model_version = mlflow.register_model(
    model_uri=model_uri,
    name=model_name
)

print("Model registered successfully!")
print("Model Name:", model_name)
print("Model Version:", model_version.version)
