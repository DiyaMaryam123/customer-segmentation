import mlflow
import numpy as np

mlflow.set_tracking_uri("sqlite:///mlflow.db")

model_uri = "models:/Customer_Segmentation_KMeans/1"

model = mlflow.sklearn.load_model(model_uri)

print("Registered model loaded successfully!")

print("Model:", model)

print("Cluster centers:")
print(model.cluster_centers_)

# Model expects exactly 2 features
sample = np.array([[0.0, 0.0]])

prediction = model.predict(sample)

print("Prediction:", prediction)
print("Assigned Cluster:", prediction[0])
