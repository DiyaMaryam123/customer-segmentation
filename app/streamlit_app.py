import streamlit as st
import requests
import os

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
.stApp {
    background-color: #f4f7fb;
}

.block-container {
    padding-top: 2rem;
    max-width: 1100px;
}

.hero {
    background: linear-gradient(120deg, #172554, #2563a6);
    padding: 35px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 36px;
    margin-bottom: 8px;
}

.hero p {
    color: #e0edff;
    font-size: 16px;
}

.section-title {
    color: #172554;
    font-size: 23px;
    font-weight: 700;
    margin-top: 20px;
}

.result-card {
    background: white;
    padding: 25px;
    border-radius: 15px;
    border-left: 6px solid #2563a6;
    box-shadow: 0 3px 12px rgba(0,0,0,0.06);
}

.segment-name {
    color: #172554;
    font-size: 25px;
    font-weight: 700;
}

.segment-description {
    color: #526174;
    font-size: 15px;
}

.footer {
    text-align: center;
    color: #718096;
    padding: 25px;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- CLUSTER PROFILES ----------------

cluster_profiles = {
    0: {
        "name": "Average Customers",
        "description": "Customers with average income and average spending.",
        "income": 55.30,
        "spending": 49.52
    },
    1: {
        "name": "Premium Customers",
        "description": "Customers with high income and high spending.",
        "income": 86.54,
        "spending": 82.13
    },
    2: {
        "name": "Low-income, High-spending Customers",
        "description": "Customers with low income and high spending.",
        "income": 25.73,
        "spending": 79.36
    },
    3: {
        "name": "High-income, Low-spending Customers",
        "description": "Customers with high income and low spending.",
        "income": 88.20,
        "spending": 17.11
    },
    4: {
        "name": "Low-income, Low-spending Customers",
        "description": "Customers with low income and low spending.",
        "income": 26.30,
        "spending": 20.91
    }
}

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000/predict"
)

# ---------------- HEADER ----------------

st.markdown("""
<div class="hero">
    <h1>📊 Customer Segmentation Dashboard</h1>
    <p>Understand customer behaviour using Machine Learning
    and K-Means Clustering.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- INPUT SECTION ----------------

st.markdown('<div class="section-title">Customer Information</div>',
            unsafe_allow_html=True)

input_col, info_col = st.columns([1.4, 1])

with input_col:
    with st.container(border=True):
        st.subheader("Enter Customer Details")

        income = st.number_input(
            "Annual Income (in thousands)",
            min_value=1.0,
            max_value=200.0,
            value=60.0,
            step=1.0
        )

        spending_score = st.slider(
            "Spending Score",
            min_value=1,
            max_value=100,
            value=50
        )

        predict_button = st.button(
            "🔍 Predict Customer Segment",
            use_container_width=True,
            type="primary"
        )

with info_col:
    with st.container(border=True):
        st.subheader("About This Model")
        st.write("**Algorithm:** K-Means Clustering")
        st.write("**Number of clusters:** 5")
        st.write("**Input features:** Annual Income and Spending Score")
        st.write("**Silhouette Score:** 0.5547")

        st.info(
            "The model groups customers according to "
            "similar income and spending behaviour."
        )

# ---------------- PREDICTION ----------------

if predict_button:

    customer_data = {
        "Income": income,
        "SpendingScore": spending_score
    }

    try:
        response = requests.post(
            API_URL,
            json=customer_data,
            timeout=10
        )

        if response.status_code == 200:

            result = response.json()
            cluster = result["assigned_cluster"]
            profile = cluster_profiles[cluster]

            st.markdown(
                '<div class="section-title">Segmentation Result</div>',
                unsafe_allow_html=True
            )

            st.success("Customer segment predicted successfully!")

            with st.container(border=True):

                col1, col2 = st.columns(2)

                col1.metric("Assigned Cluster", cluster)
                col2.metric("Annual Income", f"{income:.2f} thousand")

                st.markdown(
                    f'<div class="segment-name">{profile["name"]}</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="segment-description">'
                    f'{profile["description"]}</div>',
                    unsafe_allow_html=True
                )

                st.divider()

                st.subheader("Cluster Profile")

                metric1, metric2 = st.columns(2)

                metric1.metric(
                    "Average Annual Income",
                    f'{profile["income"]:.2f} thousand'
                )

                metric2.metric(
                    "Average Spending Score",
                    f'{profile["spending"]:.2f}'
                )

        else:
            st.error(f"Prediction failed: {response.text}")

    except requests.exceptions.RequestException:
        st.error(
            "Unable to connect to the prediction API. "
            "Please check whether FastAPI is running."
        )

# ---------------- FOOTER ----------------

st.markdown("""
<div class="footer">
    Customer Segmentation Project | K-Means Clustering | Machine Learning
</div>
""", unsafe_allow_html=True)
