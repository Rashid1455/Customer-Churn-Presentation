# Customer data

Supply a CSV, TSV, JSON, XLSX, or XLS workbook through the app. JSON should be
an array of objects, with one object per customer. CSV delimiters are detected
automatically; text files should use UTF-8 encoding. A local file named
`Telco_customer_churn.xlsx` is detected automatically when present.
Customer data, generated reports, and trained models are excluded from Git.
The existing local workbook is not redistributed because its license and
provenance have not been established in this repository.

Training needs at least 40 unique rows, at least 8 examples per class, and
two supported predictors. Use one row per customer.

| Field | Example | Required |
| --- | --- | --- |
| CustomerID | CUSTOMER-001 | Optional identifier |
| Churn | Yes or No | Training only; Churn Value (0/1) also accepted |
| tenure | 12 | At least two predictors |
| MonthlyCharges | 69.50 | At least two predictors |
| Contract | Month-to-month | Optional predictor |

See `FEATURES` in `churn_model.py` for all supported predictors. Prediction
files must include every predictor selected during training. Do not upload
serialized Python models from untrusted sources.
