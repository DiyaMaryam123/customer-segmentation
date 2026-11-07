import os
import logging
from collections import Counter

import joblib
import pandas as pd

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

# ---------------- PATHS ----------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(LOG_DIR, exist_ok=True)

# ---------------- LOGGING ----------------

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "api.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# ---------------- LOAD MODEL ----------------

model = joblib.load(
    os.path.join(BASE_DIR, "models", "kmeans_model.pkl")
)

scaler = joblib.load(
    os.path.join(BASE_DIR, "models", "scaler.pkl")
)

# ---------------- APP ----------------

app = FastAPI(
    title="Customer Segmentation API",
    description="API for predicting customer segments using K-Means",
    version="1.0"
)

# ---------------- MONITORING COUNTERS ----------------

assignment_counts = Counter()
failure_count = 0

# ---------------- REQUEST MONITORING ----------------


@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    global failure_count

    response = await call_next(request)

    if response.status_code >= 400 and request.url.path != "/favicon.ico":
        failure_count += 1
        logger.error(
            "Request failed: %s %s | Status: %s",
            request.method,
            request.url.path,
            response.status_code
        )

    return response
# ---------------- INPUT VALIDATION ----------------

class CustomerData(BaseModel):
    Income: float = Field(gt=0, le=200)
    SpendingScore: float = Field(gt=0, le=100)

# ---------------- ROUTES ----------------

@app.get("/")
def home():
    return {
        "message": "Customer Segmentation API is running!",
        "model": "KMeans",
        "clusters": 5
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict_customer(data: CustomerData):

    try:
        input_data = pd.DataFrame(
            [[data.Income, data.SpendingScore]],
            columns=["Income", "SpendingScore"]
        )

        scaled_data = scaler.transform(input_data)

        prediction = model.predict(scaled_data)

        cluster = int(prediction[0])

        assignment_counts[cluster] += 1

        logger.info(
            "Prediction successful | Income: %s | Spending Score: %s | Cluster: %s",
            data.Income,
            data.SpendingScore,
            cluster
        )

        return {
            "Income": data.Income,
            "SpendingScore": data.SpendingScore,
            "assigned_cluster": cluster
        }

    except Exception:
        logger.exception("Prediction processing failed")
        raise

# ---------------- MONITORING ENDPOINT ----------------

@app.get("/metrics")
def metrics():
    return {
        "total_successful_predictions": sum(assignment_counts.values()),
        "cluster_assignment_counts": dict(assignment_counts),
        "total_failed_requests": failure_count
    }
