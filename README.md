![Retain ? Customer churn intelligence](docs/assets/banner.svg)

<p align="center">
  <strong>Python 3.11+ &nbsp; ? &nbsp; Streamlit &nbsp; ? &nbsp; Scikit-learn</strong><br>
  <a href="#quick-start">Quick start</a> &nbsp; / &nbsp;
  <a href="#your-data">Your data</a> &nbsp; / &nbsp;
  <a href="docs/GUIDE.md">Full guide</a>
</p>

## Turn customer data into a clearer retention plan

**Retain** is an interactive Telco churn dashboard. Upload historical customer
data, compare four machine learning models automatically, and score current
customers to help prioritize outreach.

### One workspace. Three useful views.

| Explore | Evaluate | Prioritize |
| :--- | :--- | :--- |
| Customer totals and churn trends | Compare four classifiers | Score current customers |
| Segment charts and missing values | Review separate test results | Search and filter by risk |
| Preview your uploaded data | Inspect model signals | Download an outreach list |

A teal interface, animated background, and clear summary cards bring the results together.

## Quick start

Install **Python 3.11 or newer**, then open a terminal in this project folder.

**Windows PowerShell**

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run Customer_prediction.py
```

<details>
<summary>macOS / Linux commands</summary>

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run Customer_prediction.py
```

</details>

Open the address printed in your terminal, usually **http://localhost:8501**.

**Upload ? Review results ? Score current customers ? Export**

Choose a labeled file on the start screen. Analysis begins automatically.
For new customers without churn labels, use **Customer scoring** after training.

## Your data

**Excel ? CSV ? TSV ? JSON** ? up to **25 MB** per file.

Training requires **40 or more unique rows**, with at least **8 churned** and
**8 retained** customers, plus **two supported predictors**.

| CustomerID | tenure | MonthlyCharges | Contract | Churn |
| :--- | ---: | ---: | :--- | :--- |
| CUSTOMER-001 | 12 | 65.00 | Month-to-month | Yes |
| CUSTOMER-002 | 36 | 49.00 | One year | No |

*Example format only. Add your own rows to meet the training requirements.*

Download a column template from the app, or read the [data guide](data/README.md).
If you already have `Telco_customer_churn.xlsx` locally, choose **Local dataset**.

## Read the results

| Low risk | Medium risk | High risk |
| :--- | :--- | :--- |
| Below 30% | 30% to below 70% | 70% or above |
| Regular customer support | Check satisfaction | Review service issues |

Models are compared using a **60% training / 20% validation / 20% test** split.
The winner is selected by validation ROC AUC; the dashboard reports separate test
results. A final model is refitted on all labeled rows for customer scoring.

> **Research prototype:** Risk scores are uncalibrated estimates, not guarantees
> or a forecast of when a customer will leave. Feature importance shows associations,
> not causes. Use held-out test results to assess quality, not fitted historical scores.

## Inside the project

```text
Customer_prediction.py       Dashboard and upload workflow
churn_model.py               Data preparation and machine learning
tests/                       Automated checks
data/                        Input data guide
docs/                        Detailed guide and project banner
.streamlit/config.toml       Theme and app settings
.github/workflows/tests.yml  GitHub Actions checks
```

Run checks from your activated Python environment:

```bash
python -m unittest discover -s tests -v
```

Tests use synthetic data; no customer workbook is required.
CI is configured for Python **3.11, 3.12, and 3.13**.

---

**Need help?** [Setup and troubleshooting](docs/GUIDE.md#troubleshooting) ?
[Model explanations](docs/GUIDE.md#explore-your-results) ?
[Publishing instructions](docs/GUIDE.md#publishing-and-deployment)

Customer workbooks, saved models, and generated reports are excluded from Git.
Keep customer JSON files in `uploads/`. Dataset redistribution rights are unverified;
a software license has not yet been selected.
