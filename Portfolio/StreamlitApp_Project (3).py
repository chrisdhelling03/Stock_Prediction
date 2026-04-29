import os, sys, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import posixpath

import joblib
import tarfile
import tempfile

import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import NumpyDeserializer

from sklearn.pipeline import Pipeline
import shap

from joblib import dump
from joblib import load


# ── Setup & Path Configuration ────────────────────────────────────────────────
warnings.simplefilter("ignore")

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import custom classes so the pipeline can be unpickled correctly
import src.Custom_Classes
import importlib
importlib.reload(src.Custom_Classes)
from src.Custom_Classes import (
    DropHighMissingCols,
    TransactionFeatureEngineer,
    DropHighCorrelation,
    SafeLabelEncoder,
    MedianImputer,
    AutoPowerTransformer,
    FeatureSelector,
)

# Reference dataset saved during notebook training
file_path = os.path.join(current_dir, 'X_train.csv')
dataset   = pd.read_csv(file_path)
dataset   = dataset.drop(columns=[c for c in dataset.columns
                                   if c.startswith('Unnamed')], errors='ignore')

# ── AWS Secrets ───────────────────────────────────────────────────────────────
aws_id       = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret   = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token    = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket   = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

# ── AWS Session ───────────────────────────────────────────────────────────────
@st.cache_resource
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name='us-east-1'
    )

session    = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)

# ── Model Configuration ───────────────────────────────────────────────────────
MODEL_INFO = {
    "endpoint"  : aws_endpoint,
    "explainer" : "explainer_sentiment.shap",
    "pipeline"  : "finalized_fraud_model.tar.gz",
    "keys"      : ['TransactionAmt', 'card6_freq_enc', 'card3', 'C12'],
    "inputs"    : [
        {"name": "TransactionAmt",  "min": 0.0,   "max": 5000.0, "default": 100.0, "step": 1.0},
        {"name": "card6_freq_enc",  "min": 0.0,   "max": 1.0,    "default": 0.5,   "step": 0.01},
        {"name": "card3",           "min": 100.0, "max": 231.0,  "default": 150.0, "step": 1.0},
        {"name": "C12",             "min": 0.0,   "max": 3000.0, "default": 1.0,   "step": 1.0},
    ]
}


# ── Load pipeline from S3 ─────────────────────────────────────────────────────
def load_pipeline(_session, bucket, key):
    s3_client = _session.client('s3')
    filename  = MODEL_INFO["pipeline"]

    s3_client.download_file(
        Filename=filename,
        Bucket=bucket,
        Key=f"{key}/{os.path.basename(filename)}")

    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=".")
        joblib_file = [f for f in tar.getnames() if f.endswith('.joblib')][0]

    return joblib.load(f"{joblib_file}")


# ── Load SHAP explainer from S3 ───────────────────────────────────────────────
def load_shap_explainer(_session, bucket, key, local_path):
    s3_client = _session.client('s3')

    if not os.path.exists(local_path):
        s3_client.download_file(Filename=local_path, Bucket=bucket, Key=key)

    with open(local_path, "rb") as f:
        return load(f)


# ── Prediction Logic ──────────────────────────────────────────────────────────
def call_model_api(input_df):
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=JSONSerializer(),
        deserializer=NumpyDeserializer()
    )
    try:
        raw_pred = predictor.predict(input_df)
        pred_val = pd.DataFrame(raw_pred).values[-1][0]
        mapping  = {0: "Legitimate", 1: "Fraud"}
        return mapping.get(int(pred_val), "Unknown"), 200
    except Exception as e:
        return f"Error: {str(e)}", 500


# ── Local Explainability ──────────────────────────────────────────────────────
def display_explanation(input_df, session, aws_bucket):
    explainer_name = MODEL_INFO["explainer"]
    explainer = load_shap_explainer(
        session, aws_bucket,
        posixpath.join('explainer', explainer_name),
        os.path.join(tempfile.gettempdir(), explainer_name)
    )

    best_pipeline          = load_pipeline(session, aws_bucket, 'sklearn-pipeline-deployment')
    preprocessing_pipeline = Pipeline(steps=best_pipeline.steps[:-2])
    input_df = pd.DataFrame([input_df])
    input_df_transformed   = preprocessing_pipeline.transform(input_df)

    # Use features_to_keep from our custom FeatureSelector (no get_support() needed)
    selected_features    = best_pipeline.named_steps['selector'].features_to_keep
    input_df_transformed = pd.DataFrame(input_df_transformed, columns=selected_features)

    shap_values = explainer(input_df_transformed, check_additivity=False)

    st.subheader("🔍 Decision Transparency (SHAP)")
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[0, :, 1])  # class 1 = Fraud
    st.pyplot(fig)

    top_feature = (
        pd.Series(shap_values[0, :, 1].values,
                  index=shap_values[0, :, 1].feature_names)
        .abs().idxmax()
    )
    st.info(f"**Business Insight:** The most influential factor in this decision was **{top_feature}**.")


# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="IEEE-CIS Fraud Detection", layout="wide")
st.title("🔐 IEEE-CIS Fraud Detection – ML Deployment")

with st.form("pred_form"):
    st.subheader("Transaction Inputs")
    cols        = st.columns(2)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp['name']] = st.number_input(
                inp['name'].replace('_', ' ').upper(),
                min_value=float(inp['min']),
                max_value=float(inp['max']),
                value=float(inp['default']),
                step=float(inp['step'])
            )

    submitted = st.form_submit_button("Run Prediction")

original = dataset.iloc[0:1].to_dict('records')[0]
original.update(user_inputs)

if submitted:
    res, status = call_model_api(original)
    if status == 200:
        st.metric("Prediction Result", res)
        display_explanation(original, session, aws_bucket)
    else:
        st.error(res)
