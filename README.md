# Customer Segmentation Using K-Means Clustering

## 1. Project Overview

This project uses K-Means clustering to segment mall customers based on their annual income and spending score.

The objective is to identify groups of customers with similar spending behaviour and provide useful insights for customer-targeted marketing strategies.

The project also demonstrates an end-to-end MLOps workflow, including experiment tracking, model registration, API development, dashboard development, Docker deployment, logging, monitoring, and testing.

## 2. Dataset

The project uses the Mall Customers dataset.
### Dataset Source and Version

- **Dataset name:** Mall Customers Dataset
- **Source platform:** Kaggle
- **Dataset type:** Customer demographic and spending data
- **Records:** 200 customers
- **Attributes:** CustomerID, Gender, Age, Annual Income, and Spending Score
- **Version:** The specific Kaggle dataset version or upload date has not yet been recorded.
- **Primary model features:** Annual Income and Spending Score.

The dataset is used for exploratory data analysis, customer segmentation, and K-Means clustering. The primary model uses Annual Income and Spending Score after applying StandardScaler.

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

## Monitoring and Maintenance

The deployed customer segmentation application uses FastAPI endpoints to monitor API health, prediction activity, and request failures.

### API Monitoring

- Use the `/health` endpoint to check API availability.
- Use the `/metrics` endpoint to review successful prediction counts, cluster assignment counts, and failed requests.
- Review API logs to investigate errors and unsuccessful requests.
- Monitor input values to identify unusual changes in customer income and spending scores.

### Model Monitoring

- Review the distribution of customer assignments across the five clusters.
- Compare new customer data distributions with the original training data.
- Investigate substantial changes in cluster proportions or input distributions.
- Evaluate whether the existing customer segments remain meaningful as new data becomes available.

### Maintenance and Retraining

- Retrain the model when sufficient representative customer data becomes available or meaningful changes in customer behaviour are observed.
- Evaluate retrained models using Silhouette Score, Davies-Bouldin Score, and Calinski-Harabasz Score.
- Compare new cluster centres and customer profiles with the existing model.
- Preserve cluster-label consistency by documenting any changes in cluster numbering.
- Version the model, scaler, metadata, evaluation results, and cluster profiles together.
- Test the updated model before deploying it.

The API's prediction counters are maintained in memory and reset when the API restarts. Log files provide additional information about API activity and failures.

### Cluster Label Consistency

K-Means cluster numbers are arbitrary and may change when the model is retrained. Do not assume that Cluster 0 or any other cluster number always represents the same customer segment. Compare the new cluster centres with the previous centres, document any label mapping, and update the cluster profiles and application descriptions when necessary.

### Limitations

- The project uses a small dataset of 200 customer records, so the segments may not represent all customer populations.
- Clustering uses only Annual Income and Spending Score; other factors that may influence customer behaviour are not included.
- Cluster assignments describe similarities in the available data and should not be treated as definitive customer classifications.
- Cluster numbers do not have an inherent meaning and may change after retraining.
- API input validation enforces the configured input ranges; values outside those ranges are rejected.

### Reproducing Training and Evaluation

The model training and evaluation workflow can be reproduced using the `kmeans.ipynb` notebook.

1. Open `kmeans.ipynb` in Google Colab or a local Jupyter environment.
2. Load the Mall Customers dataset.
3. Run the notebook cells in order to perform data exploration, preprocessing, feature scaling, and K-Means clustering.
4. Evaluate the clustering results using Silhouette Score, Davies-Bouldin Score, and Calinski-Harabasz Score.
5. Review the cluster visualizations and customer profiles.
6. Save the trained K-Means model and fitted StandardScaler.

The current model uses five clusters, `random_state=42`, and `n_init=10`.

#### Retraining and Cluster Label Consistency

When new customer data becomes available, retrain the model using the same preprocessing steps and feature order:

- Annual Income
- Spending Score

Evaluate the retrained model before replacing the existing model.

K-Means cluster labels are arbitrary and may change between training runs. Therefore:

- Compare the new cluster centres with the existing cluster centres.
- Match clusters based on their customer profiles and centre similarities.
- Document any changes in cluster numbering and segment descriptions.
- Update the cluster profiles and application descriptions if the segment mapping changes.
- Save and version the new model, scaler, metadata, evaluation metrics, and cluster profiles together.

Do not replace the deployed model until the new model has been evaluated and its cluster labels have been checked for consistency.
