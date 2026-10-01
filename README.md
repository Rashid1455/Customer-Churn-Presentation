<div align="center">

# Retain
### Customer churn prediction made simple

Upload customer data. Explore patterns. Find customers who may need attention.

**Python 3.11+ · Streamlit · Scikit-learn**

[Get started](#get-started) · [Prepare your data](#prepare-your-data) · [Explore results](#explore-your-results) · [Troubleshooting](#troubleshooting)

</div>

---

## What does this project do?

Retain helps you understand **customer churn** — when customers stop using a service.
It learns from historical Telco customer data, compares four machine learning models,
and estimates churn risk for current customers.

| Upload | Understand | Take action |
| :--- | :--- | :--- |
| Import Excel, CSV, TSV, or JSON | View charts, customer trends, and model results | Search customers, filter risk levels, and export a list |

## Get started

You need **Python 3.11 or newer**. Open a terminal in this project folder.

### 1. Set up your environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

<details>
<summary>Using macOS or Linux?</summary>

Activate the environment with this command instead:

```bash
source .venv/bin/activate
```

</details>

### 2. Install the packages

```bash
python -m pip install -r requirements.txt
```

### 3. Launch the app

```bash
python -m streamlit run Customer_prediction.py
```

Open the local address shown in the terminal, usually **http://localhost:8501**.

## Upload your first file

1. **Upload historical customer data** on the start screen. For Excel files, choose a worksheet.
2. **Wait for the results.** The app automatically compares four models when your data is valid.
3. **Explore the dashboard.** Use the sidebar to switch between overview, model performance, and customer scoring.
4. **Score current customers.** Open **Customer scoring**, upload their data, and download your filtered results.

> **Try the local dataset:** Choose **Local dataset** if `Telco_customer_churn.xlsx` is available in your project folder.

## Prepare your data

Supported formats: **`.xlsx` · `.xls` · `.csv` · `.tsv` · `.json`**  
Maximum upload size: **25 MB per file**

Your historical training file needs:

- **At least 40 rows**, with one row per customer.
- **At least 8 churned and 8 retained customers.**
- A **churn label**: `Churn` or `Churn Label` with `Yes` / `No`, or `Churn Value` with `0` / `1`.
- **At least two supported customer features**, such as `tenure`, `MonthlyCharges`, or `Contract`.

Example columns:

| CustomerID | tenure | MonthlyCharges | Contract | Churn |
| --- | ---: | ---: | --- | --- |
| CUSTOMER-001 | 12 | 65.00 | Month-to-month | Yes |
| CUSTOMER-002 | 36 | 49.00 | One year | No |

*These two rows illustrate the format; they are not enough to train a model.*

For new-customer scoring, include the same feature columns used during training.
**Churn labels are not required for scoring.** Train on labeled historical data first.

Use UTF-8 for text files. JSON should contain an array of customer objects.
You can download a column template from the app's start screen.
See the [data guide](data/README.md) for more details and `FEATURES` in
[churn_model.py](churn_model.py) for all supported predictors.

## Explore your results

| Workspace | What you will see |
| --- | --- |
| **Overview** | Customer totals, observed churn, segment charts, and missing values |
| **Model performance** | Model comparison, test scores, prediction errors, and leading model signals |
| **Customer scoring** | Customer search, estimated churn probabilities, risk filters, and CSV download |

### Understand the risk levels

| Risk level | Estimated churn probability | Suggested next step |
| --- | --- | --- |
| Low | Below 30% | Maintain regular customer support |
| Medium | 30% to below 70% | Check satisfaction and plan suitability |
| High | 70% or above | Review service issues and consider retention options |

The sidebar's **classification threshold** controls when a customer is classified
as likely to churn. It does not change these fixed risk bands.

<details>
<summary><strong>What do the model scores mean?</strong></summary>

| Score | Plain-language meaning |
| --- | --- |
| **Precision** | Of the customers flagged as likely to churn, how many actually churned? |
| **Recall** | Of the customers who actually churned, how many did the model identify? |
| **F1** | A combined measure of precision and recall |
| **ROC AUC** | How well the model ranks churned customers above retained customers; 1.0 is perfect and 0.5 is chance-level ranking |
| **Confusion matrix** | A table showing correct predictions, false alarms, and missed churn cases |

Lowering the classification threshold flags more customers. This can catch more
churn cases, but also produce more false alarms. Choose a threshold based on how
many customers your team can contact and the cost of unnecessary outreach.

</details>

> **Use results as decision support.** This is a research prototype. Probabilities
> are estimates, not guarantees, and do not specify when a customer might leave.
> Scores for training customers are not an independent measure of model quality;
> use the separate test results for that.

<details>
<summary><strong>How the models work</strong></summary>

The app compares **logistic regression, decision tree, random forest, and gradient boosting**.

- **60% training:** teaches each model customer patterns.
- **20% validation:** selects the best model using ROC AUC, a measure of how well it separates churned and retained customers.
- **20% testing:** evaluates the selected model on data held out from selection.

Missing-value handling and encoding are fitted within each training split.
Customer IDs and outcome fields such as churn reason and churn score are excluded
from model inputs. Duplicate customer IDs are rejected.

The selected model is refitted on training and validation data for test evaluation.
A separate final model is then trained on all labeled rows for customer scoring.
The random seed is 42 for reproducibility.

Probabilities are not calibrated, and risk bands are heuristics. Feature importance
shows associations, not proof that changing a feature will prevent churn.

</details>

## Project structure

```text
Customer_prediction.py       Streamlit app and dashboard
churn_model.py               Data preparation, training, and scoring
tests/                      Automated tests
data/README.md              Input data guide
.streamlit/config.toml       Theme and upload settings
.github/workflows/tests.yml  GitHub Actions checks
requirements.txt            Required Python packages
```

## Troubleshooting

| Problem | What to do |
| --- | --- |
| My file has no churn column | Train with historical data containing churn labels, then upload this file in **Customer scoring**. |
| The app says there are too few rows | Supply at least 40 unique training rows, including at least 8 customers in each churn class. |
| Prediction columns are missing | Include every predictor used during training. The error lists the missing columns. |
| Customer IDs are repeated | Keep one training row per customer and resolve conflicting records. |
| The local dataset is unavailable | Choose **Upload** and supply your own file. Customer workbooks are excluded from GitHub. |
| No customers appear after filtering | Select more risk bands or clear the customer search. |
| A Python package is missing | Activate your environment and run `python -m pip install -r requirements.txt` again. |

<details>
<summary><strong>PowerShell blocks environment activation</strong></summary>

Run the environment's Python directly, without activating it:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run Customer_prediction.py
```

</details>

<details>
<summary><strong>Port 8501 is already in use</strong></summary>

Choose another port and open the address printed in the terminal:

```bash
python -m streamlit run Customer_prediction.py --server.port 8502
```

</details>

## Run the tests

```bash
python -m unittest discover -s tests -v
```

Tests cover data validation, supported file formats, model training, scoring,
model export, automatic upload analysis, and app navigation. Tests use synthetic
data, so the local customer workbook is not needed. GitHub Actions is configured
to run checks on Python 3.11, 3.12, and 3.13.

<details>
<summary><strong>Publishing and deployment</strong></summary>

Customer files, generated reports, saved models, secrets, and local backups are
excluded by `.gitignore`. Existing local files stay on your computer.
Keep any customer JSON files in an `uploads/` folder, which is also ignored.

To publish, create an empty GitHub repository, then run:

```bash
git init
git add .
git diff --cached --stat
git commit -m "Prepare customer churn workspace"
git branch -M main
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```

Replace `YOUR_REPOSITORY_URL` with your repository's URL. Review the staged files
before committing. Dataset redistribution rights have not been verified, and no
software license has been selected. Add your preferred license before inviting reuse.

For Streamlit hosting, use `Customer_prediction.py` as the entry point. Upload
training data after opening the deployed app.

Downloaded joblib bundles contain `model`, `features`, and `name`. Load only
trusted bundles with compatible Python and scikit-learn versions.

</details>
