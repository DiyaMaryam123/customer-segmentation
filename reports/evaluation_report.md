# Customer Segmentation Using K-Means
## Model Evaluation Report

### 1. Objective

The objective of this project is to group mall customers into meaningful segments using K-Means clustering based on annual income and spending score.

### 2. Dataset Overview

The Mall Customers dataset contains 200 customer records and five attributes.

- Total records: 200
- Missing values: 0
- Duplicate records: 0
- Features used for primary clustering: Annual Income and Spending Score

### 3. Data Preprocessing

The primary clustering workflow used Annual Income and Spending Score.

StandardScaler was applied to standardize the two features before model training.

### 4. Model Configuration

- Algorithm: K-Means
- Number of clusters: 5
- Random state: 42
- Number of initializations: 10
- Input features: Annual Income and Spending Score

### 5. Model Evaluation

| Evaluation Metric | Result |
|---|---:|
| Silhouette Score | 0.5547 |
| Davies-Bouldin Index | 0.5722 |
| Calinski-Harabasz Score | 248.6493 |

The selected five-cluster model achieved a Silhouette Score of 0.5547. The evaluation results support the use of five clusters for this dataset and feature selection.

### 6. Cluster Profiles

| Cluster | Average Annual Income | Average Spending Score | Interpretation |
|---|---:|---:|---|
| 0 | 55.30 | 49.52 | Average Customers |
| 1 | 86.54 | 82.13 | Premium Customers |
| 2 | 25.73 | 79.36 | High-Spending, Low-Income Customers |
| 3 | 88.20 | 17.11 | High-Income, Low-Spending Customers |
| 4 | 26.30 | 20.91 | Low-Income, Low-Spending Customers |

Annual income values are in thousands.

### 7. PCA-Assisted Comparison

A separate PCA-assisted workflow was explored using Age, Annual Income, and Spending Score.

The three input features were standardized before applying PCA with two components.

- Explained variance: 77.57%
- Number of clusters: 5
- Silhouette Score: 0.3931
- Davies-Bouldin Index: 0.8045

This workflow provides a two-dimensional representation for visualization. Its scores should not be directly compared with the primary model because the two workflows use different feature spaces.

### 8. MLOps and Deployment Evaluation

The project includes:

- MLflow experiment tracking and model registration.
- Saved K-Means model and StandardScaler artifacts.
- FastAPI prediction endpoint.
- Streamlit customer segmentation dashboard.
- Docker-based application deployment.
- API logging and prediction monitoring.
- Input validation for customer attributes.

### 9. API Testing Results

Five automated API tests passed:

1. Home endpoint test.
2. Health endpoint test.
3. Valid customer prediction test.
4. Invalid income validation test.
5. Invalid spending score validation test.

Test result: **5 passed, 0 failed.**

### 10. Conclusion

The K-Means model successfully groups customers into five segments using annual income and spending score.

The project demonstrates the complete workflow from data preprocessing and model evaluation to experiment tracking, API development, dashboard integration, containerization, monitoring, and automated testing.

The resulting customer segments can help businesses understand different spending patterns and support targeted marketing strategies.

## Monitoring and Maintenance

The deployed API provides `/health` and `/metrics` endpoints to check service health, successful assignment counts, cluster assignment counts, and failed requests. Incoming customer feature distributions and cluster assignment proportions should be reviewed for changes from the training data.

Retraining should be considered when representative new customer data becomes available or when meaningful changes in customer behaviour are observed. A retrained model should be evaluated using silhouette score and Davies-Bouldin score, and its cluster profiles should be reviewed against the existing model before deployment.

## Cluster Label Consistency

K-Means cluster numbers are arbitrary. Cluster IDs may change after retraining, even when the underlying customer segments are similar. New cluster centres should therefore be compared with the previous centres, and any mapping between old and new labels should be documented. Application profiles and descriptions must be updated when the mapping changes.

## Limitations

- The dataset contains 200 customer records and may not represent broader customer populations.
- The model uses only Annual Income and Spending Score.
- The resulting clusters describe patterns in the available data; they are not definitive customer classifications.
- Cluster IDs do not have a permanent semantic meaning and may change after retraining.
- The API rejects values outside its configured input ranges.
