"""Interactive Telco retention workspace. Run with Streamlit."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix

from churn_model import read_data, training_data, train_models, metrics, score_customers

st.set_page_config(page_title="Retain | Customer intelligence", page_icon=":material/insights:", layout="wide")

st.html("""<style>
/* Decorative layers stay behind content and never intercept input. */
[data-testid="stAppViewContainer"] {isolation: isolate;}
[data-testid="stAppViewContainer"]::before,
[data-testid="stAppViewContainer"]::after {
    content: ""; position: fixed; inset: -15%; z-index: -1;
    pointer-events: none;
    background: radial-gradient(ellipse at 15% 25%, #14b8a624, transparent 42%),
                radial-gradient(ellipse at 85% 65%, #3b82f620, transparent 45%);
    animation: retain-drift 24s ease-in-out infinite alternate;
}
[data-testid="stAppViewContainer"]::after {
    background-image:
        radial-gradient(circle, #0d948866 3px, transparent 4px),
        radial-gradient(circle, #3b82f64d 5px, transparent 6px),
        radial-gradient(circle, #8b5cf64d 2px, transparent 3px);
    background-size: 180px 180px, 310px 310px, 130px 130px;
    background-position: 0 0, 60px 90px, 30px 40px;
    animation: retain-particles 12s linear infinite;
}
@keyframes retain-particles {
    to {background-position: 180px -180px, -250px -220px, 160px -90px;}
}
@keyframes retain-orbit {
    from {transform: translate3d(0, -25px, 0) rotate(0deg);}
    to {transform: translate3d(-100px, 35px, 0) rotate(35deg);}
}
@keyframes retain-drift {
    from {transform: translate3d(-2%, -1%, 0) scale(1);}
    to {transform: translate3d(3%, 4%, 0) scale(1.06);}
}
.st-key-hero {background: linear-gradient(115deg, #102c46, #115e59); padding: 2rem;
border-radius: 20px; margin-bottom: 1.2rem; position: relative;
overflow: hidden; isolation: isolate;}
.st-key-hero::before {
    content: ""; position: absolute; inset: -60%; z-index: -1; pointer-events: none;
    background: radial-gradient(ellipse at 65% 50%, #2dd4bf38, transparent 35%),
                repeating-linear-gradient(120deg, transparent 0 70px, #ffffff08 71px 72px, transparent 73px 140px);
    animation: retain-drift 20s ease-in-out infinite alternate;
}
.st-key-hero::after {
    content: ""; position: absolute; right: -55px; top: -65px;
    width: 260px; height: 260px; border: 2px solid #5eead480;
    border-radius: 42%; pointer-events: none; z-index: -1;
    box-shadow: 0 0 0 30px #5eead412, 0 0 0 65px #5eead410,
                0 0 0 100px #5eead40a;
    animation: retain-orbit 7s ease-in-out infinite alternate;
}
.st-key-hero h1, .st-key-hero p {color: #ffffff !important;}
.st-key-hero h1 {font-size: clamp(2rem, 4vw, 3.4rem); letter-spacing: -0.04em;}
.st-key-upload_panel {border: 1px solid #cbdedc; border-radius: 16px; padding: 1.4rem;
background: #ffffffed;}
@media (prefers-reduced-motion: reduce) {
    [data-testid="stAppViewContainer"]::before,
    [data-testid="stAppViewContainer"]::after,
    .st-key-hero::before, .st-key-hero::after {animation: none !important;}
}
</style>""")

@st.cache_data(max_entries=4, ttl=3600)
def load_table(content, name, sheet):
    return read_data(content, name, sheet)


def upload_table(label, key):
    uploaded = st.file_uploader(label, type=["csv", "xlsx", "xls", "tsv", "json"], key=key,
                                help="Excel workbooks, delimited tables, or JSON arrays of records. Up to 25 MB.")
    if uploaded is None:
        return None, None
    content = uploaded.getvalue()
    sheet = 0
    if uploaded.name.lower().endswith((".xlsx", ".xls")):
        with pd.ExcelFile(BytesIO(content)) as workbook:
            sheet = st.selectbox("Worksheet", workbook.sheet_names, key=f"{key}_sheet")
    st.caption(f"{uploaded.name} · {len(content) / 1024:,.0f} KB")
    return load_table(content, uploaded.name, sheet), sha256(content + uploaded.name.encode() + str(sheet).encode()).hexdigest()


with st.sidebar:
    st.title(":material/insights: Retain")
    st.caption("CUSTOMER INTELLIGENCE")
    view = st.radio("Workspace", ["Overview", "Model performance", "Customer scoring"], key="view")
    st.subheader("Decision settings")
    threshold = st.slider("Classification threshold", 0.05, 0.95, 0.5, 0.05,
                          help="A score at or above this value is classified as churn. Risk bands stay fixed.")
    st.caption("Low <30% · Medium 30–70% · High ≥70%")
    st.caption("Reproducible training · Seed 42")
    st.caption("Created by Rashid Ali")
    st.link_button("Connect on LinkedIn", "https://www.linkedin.com/in/rashid-ali-619671357/",
                   icon=":material/open_in_new:", width="stretch")

with st.container(key="hero"):
    st.caption("RETAIN / CUSTOMER INTELLIGENCE")
    st.title("Your data. Clearer decisions.")
    st.write("Upload your customer data and turn churn patterns into a focused retention plan.")
    st.caption("01  Upload your file     /     02  Explore results     /     03  Prioritize outreach")

with st.container(key="upload_panel"):
    st.subheader(":material/upload_file: Start with your customer data")
    st.caption("Excel, CSV, TSV, or JSON · Results appear automatically after a valid upload")
    source = st.segmented_control("Data source", ["Upload", "Local dataset"], default="Upload")
    try:
        if source == "Upload":
            raw, fingerprint = upload_table("Upload labeled customer data", "training_upload")
        else:
            path = Path(__file__).with_name("Telco_customer_churn.xlsx")
            if path.exists():
                content = path.read_bytes()
                raw = load_table(content, path.name, 0)
                fingerprint = sha256(content).hexdigest()
            else:
                raw, fingerprint = None, None
                st.info("Upload a labeled CSV or Excel file to get started. See data/README.md for the schema.")
        if fingerprint != st.session_state.get("fingerprint"):
            st.session_state.pop("result", None)
            st.session_state["fingerprint"] = fingerprint
        if raw is None:
            st.info("Drop a labeled customer file above, or choose Local dataset to explore the included workbook.")
            with st.expander("What should my file contain?", expanded=True):
                st.write("Include a Churn column with Yes/No (or Churn Value with 0/1), at least two customer predictors, and at least 40 rows with 8 customers in each class.")
                st.caption("Examples: tenure, MonthlyCharges, Contract. For customers without churn labels, first train on historical data, then use Customer scoring.")
                template = pd.DataFrame({"CustomerID": ["CUSTOMER-001", "CUSTOMER-002"], "tenure": [12, 36], "MonthlyCharges": [65, 49], "Contract": ["Month-to-month", "One year"], "Churn": ["Yes", "No"]})
                st.download_button("Download column template", template.to_csv(index=False), "customer_template.csv", "text/csv")
                st.caption("The template illustrates the columns; replace its examples with your own training rows.")
            cols = st.columns(3)
            for col, title, description in zip(cols, ["Explore your customers", "Measure model quality", "Plan your outreach"], ["See customer trends and observed churn in one place.", "Compare four models with a separate test set.", "Sort customers by risk and export your results."]):
                with col.container(border=True):
                    st.subheader(title)
                    st.write(description)
            st.stop()
        data, features = training_data(raw)
        st.caption(f"{len(data):,} unique rows · {len(features)} eligible predictors · Outcome columns excluded")
        if "result" not in st.session_state:
            with st.spinner("Comparing four models and evaluating the selected model…"):
                st.session_state.result = train_models(data, features)
            st.success("Your results are ready. Explore the overview below or choose a workspace from the sidebar.")
    except (ValueError, OSError, ImportError, KeyError) as exc:
        st.session_state.pop("result", None)
        st.error(f"Unable to use this data: {exc}")
        st.stop()

result = st.session_state.get("result")
if view == "Overview":
    with st.container(horizontal=True):
        st.metric("Customers", f"{len(data):,}", border=True)
        st.metric("Observed churn", f"{data.churn_value.mean():.1%}", border=True)
        st.metric("Predictors", str(len(features)), border=True)
        st.metric("Test ROC AUC", f"{metrics(result['test_y'], result['test_probability'])['ROC AUC']:.3f}", border=True)
    st.subheader("Your dataset at a glance")
    st.caption(f"{int(data.churn_value.sum()):,} customers have churned in this dataset. {result['name']} performed best on validation ROC AUC.")
    left, right = st.columns(2)
    with left.container(border=True):
        st.subheader("Customer outcomes")
        counts = data.churn_value.map({0: "Retained", 1: "Churned"}).value_counts()
        st.bar_chart(counts, color="#0F766E", height=280)
    with right.container(border=True):
        st.subheader("Churn by customer segment")
        segments = [c for c in ["contract", "internet_service", "payment_method"] if c in data]
        if segments:
            segment = st.selectbox("Segment", segments, format_func=lambda s: s.replace("_", " ").capitalize())
            summary = data.groupby(segment).churn_value.agg(["mean", "count"])
            summary.columns = ["Churn rate", "Customers"]
            st.bar_chart(summary["Churn rate"], color="#2563EB", height=190)
            st.dataframe(summary, column_config={"Churn rate": st.column_config.NumberColumn(format="percent")})
        else:
            st.caption("Include Contract, Internet Service, or Payment Method to compare segments.")
    with st.expander("Preview training data and missing values"):
        st.dataframe(data.head(100), hide_index=True)
        st.dataframe(data[features].isna().sum().rename("Missing values"))
    if not result:
        st.info("Next: open Training data and select Train and compare models.")

elif result is None:
    st.info("Train a model first using the Training data panel above.")

elif view == "Model performance":
    st.subheader(result["name"])
    st.caption("Selected by validation ROC AUC. The metrics below use the separate test set.")
    scores = metrics(result["test_y"], result["test_probability"], threshold)
    with st.container(horizontal=True):
        for name in ["ROC AUC", "Precision", "Recall", "F1"]:
            st.metric(name, f"{scores[name]:.3f}", border=True)
    left, right = st.columns(2)
    with left.container(border=True):
        st.subheader("Validation comparison")
        st.dataframe(result["comparison"].round(3), hide_index=True)
        st.caption("Comparison uses a fixed 0.50 threshold; ROC AUC is threshold independent.")
    with right.container(border=True):
        st.subheader("Test confusion matrix")
        matrix = confusion_matrix(result["test_y"], result["test_probability"] >= threshold, labels=[0, 1])
        st.dataframe(pd.DataFrame(matrix, index=["Actual retained", "Actual churn"], columns=["Predicted retained", "Predicted churn"]))
        st.caption(f"Train / validation / test rows: {result['split_sizes']}")
    st.subheader("Leading model signals")
    st.bar_chart(result["drivers"].head(12), x="Feature", y="Importance", horizontal=True, color="#0F766E")
    st.caption("Importance reflects model associations, not causal effects. Coefficients and tree importance have different scales.")
    model_buffer = BytesIO()
    joblib.dump({"model": result["model"], "features": result["features"], "name": result["name"]}, model_buffer)
    with st.container(horizontal=True):
        st.download_button("Download model bundle", model_buffer.getvalue(), "churn_model.joblib")
        st.download_button("Download comparison", result["comparison"].to_csv(index=False), "model_comparison.csv", "text/csv")

elif view == "Customer scoring":
    st.subheader("Build your outreach list")
    st.write("Upload current customers with the same predictors used for training. Churn labels are optional.")
    try:
        current, _ = upload_table("Customer file", "scoring_upload")
        historical = st.toggle("Explore fitted historical scores", value=False,
                               help="Uses training customers. These scores are not an unbiased evaluation.")
        if current is None and historical:
            current = data
            st.warning("Historical fitted scores include customers already seen during training. Use test metrics to assess quality.")
        if current is not None:
            scored = score_customers(current, result, threshold)
            with st.container(horizontal=True):
                for risk in ["High", "Medium", "Low"]:
                    st.metric(f"{risk} risk", f"{scored.risk_level.eq(risk).sum():,}", border=True)
            bands = st.multiselect("Risk bands", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
            filtered = scored[scored.risk_level.isin(bands)]
            if "customer_id" in filtered:
                query = st.text_input("Find a customer", placeholder="Search customer ID")
                if query:
                    filtered = filtered[filtered.customer_id.astype(str).str.contains(query, case=False, regex=False)]
            st.caption(f"{len(filtered):,} customers · Highest probability first")
            leading = [c for c in ["customer_id", "risk_level", "churn_probability", "suggested_action"] if c in filtered]
            st.dataframe(filtered, hide_index=True, column_order=leading + [c for c in filtered if c not in leading], column_config={"churn_probability": st.column_config.ProgressColumn("Churn probability", min_value=0, max_value=1, format="percent")})
            st.download_button("Download filtered outreach list", filtered.to_csv(index=False), "customer_risk_scores.csv", "text/csv", icon=":material/download:")
    except (ValueError, OSError, ImportError, KeyError) as exc:
        st.error(f"Unable to score this file: {exc}")

st.caption("Research prototype · Probabilities are not calibrated · No future prediction horizon is established by this dataset.")
