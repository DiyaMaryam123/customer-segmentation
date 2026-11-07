# Customer Segmentation Using K-Means Clustering

## 1. Project Overview

This project uses K-Means clustering to segment mall customers based on their annual income and spending score.

The objective is to identify groups of customers with similar spending behaviour and provide useful insights for customer-targeted marketing strategies.

The project also demonstrates an end-to-end MLOps workflow, including experiment tracking, model registration, API development, dashboard development, Docker deployment, logging, monitoring, and testing.

## 2. Dataset

The project uses the Mall Customers dataset.

Important attributes:
- CustomerID
- Gender
- Age
- Annual Income
- Spending Score

The primary clustering model uses:
- Annual Income
- Spending Score

## 3. Model Development

Algorithm: K-Means Clustering

Number of clusters: 5

Random state: 42

Number of initializations: 10

Preprocessing: StandardScaler

### Model Evaluation

| Metric | Result |
|---|---:|
| Silhouette Score | 0.5547 |
| Davies-Bouldin Score | 0.5722 |
| Calinski-Harabasz Score | 248.6493 |

The selected model uses five clusters based on the clustering evaluation results.

A PCA-assisted workflow was also explored for visualization and comparison.

## 4. Customer Segments

| Cluster | Segment | Average Income | Average Spending Score |
|---|---|---:|---:|
| 0 | Average Customers | 55.30 | 49.52 |
| 1 | Premium Customers | 86.54 | 82.13 |
| 2 | High-Spending, Low-Income Customers | 25.73 | 79.36 |
| 3 | High-Income, Low-Spending Customers | 88.20 | 17.11 |
| 4 | Low-Income, Low-Spending Customers | 26.30 | 20.91 |

Income values are in thousands.

## 5. MLOps Implementation

The project includes:

- MLflow experiment tracking.
- Model registration.
- Saved K-Means model and scaler.
- Model metadata and evaluation metrics.
- API logging and prediction monitoring.
- Input validation.
- Automated API tests.

## 6. Application Architecture

The application contains:

- FastAPI backend for customer segment prediction.
- Streamlit dashboard for entering customer details and viewing predictions.
- Docker containers for deployment.

## 7. Project Structure

```text
customer-segmentation/
├── api/
│   └── main.py
├── app/
│   └── streamlit_app.py
├── models/
│   ├── kmeans_model.pkl
│   ├── scaler.pkl
│   └── model_metadata.json
├── outputs/
│   ├── customer_segments.csv
│   └── cluster_profiles.csv
├── src/
│   ├── log_model.py
│   ├── mlflow_tracking.py
│   ├── register_model.py
│   └── test_registered_model.py
├── tests/
│   └── test_api.py
├── logs/
│   └── api.log
├── kmeans.ipynb
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 8. Running the Application

### Build Docker images

```bash
docker compose build
```

### Start the containers

```bash
docker compose up -d
```

### Check container status

```bash
docker compose ps
```

### Access the applications

- Streamlit dashboard: http://localhost:8501
- FastAPI documentation: http://localhost:8000/docs
- API health endpoint: http://localhost:8000/health
- API monitoring endpoint: http://localhost:8000/metrics

### Stop the containers

```bash
docker compose down
```

## 9. API Prediction Example

Endpoint:

`POST /predict`

Example request:

```json
{
  "Income": 60,
  "SpendingScore": 50
}
```

The API returns the assigned customer cluster.

## 10. Testing

Run the automated API tests:

```bash
python -m pytest tests/test_api.py -v
```

The current test run completed with five passing tests.

## 11. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- MLflow
- FastAPI
- Streamlit
- Docker
- Pytest
- Git

## 12. Conclusion

This project demonstrates customer segmentation using unsupervised machine learning and an MLOps deployment workflow.

The trained K-Means model identifies five customer groups based on annual income and spending score. The API and Streamlit dashboard make it possible to assign new customers to these segments.
