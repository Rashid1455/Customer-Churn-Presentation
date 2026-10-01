"""Data preparation and reproducible Telco churn modelling."""
import re
from io import BytesIO

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

FEATURES = ['gender', 'senior_citizen', 'partner', 'dependents', 'tenure_months',
            'phone_service', 'multiple_lines', 'internet_service', 'online_security',
            'online_backup', 'device_protection', 'tech_support', 'streaming_tv',
            'streaming_movies', 'contract', 'paperless_billing', 'payment_method',
            'monthly_charges', 'total_charges']
NUMERIC = ['tenure_months', 'monthly_charges', 'total_charges']
ALIASES = {re.sub('[^a-z0-9]', '', c): c for c in FEATURES + ['customer_id', 'churn_value', 'churn_label']}
ALIASES.update({'tenure': 'tenure_months', 'churn': 'churn_label'})


def read_data(content, name, sheet=0):
    if not name.lower().endswith(('.csv', '.xlsx', '.xls', '.tsv', '.json')):
        raise ValueError('Use a CSV, XLSX, XLS, TSV, or JSON file.')
    source = BytesIO(content)
    if name.lower().endswith('.json'):
        return pd.read_json(source, orient='records')
    if name.lower().endswith(('.csv', '.tsv')):
        return pd.read_csv(source, sep='\t' if name.lower().endswith('.tsv') else None,
                           engine='python', encoding='utf-8-sig')
    return pd.read_excel(source, sheet_name=sheet)


def clean_data(raw):
    df = raw.copy()
    df.columns = [ALIASES.get(re.sub('[^a-z0-9]', '', str(c).lower()),
                             re.sub(r'\W+', '_', str(c).strip().lower())) for c in df.columns]
    if df.columns.duplicated().any():
        raise ValueError('Column names become duplicates after normalization. Rename the duplicate columns.')
    for c in df.select_dtypes(include=['object', 'string']).columns:
        df[c] = df[c].map(lambda v: v.strip() if isinstance(v, str) else v).replace('', np.nan)
    for c in NUMERIC:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
    if 'senior_citizen' in df:
        df['senior_citizen'] = df['senior_citizen'].replace({0: 'No', 1: 'Yes', '0': 'No', '1': 'Yes'})
    if 'total_charges' in df and 'tenure_months' in df:
        df.loc[df.total_charges.isna() & df.tenure_months.eq(0), 'total_charges'] = 0
    return df


def training_data(raw):
    df = clean_data(raw).drop_duplicates().reset_index(drop=True)
    target = next((c for c in ['churn_value', 'churn_label'] if c in df), None)
    if target is None:
        raise ValueError('Training needs Churn Value (0/1), Churn Label (Yes/No), or Churn (Yes/No).')
    y = df[target].astype('string').str.lower().map({'0': 0, '1': 1, '0.0': 0, '1.0': 1, 'yes': 1, 'no': 0})
    if y.isna().any():
        raise ValueError('Every training row needs a valid churn label: Yes/No or 0/1.')
    if len(y) < 40 or y.value_counts().min() < 8 or y.nunique() != 2:
        raise ValueError('Provide at least 40 rows with at least 8 customers in each churn class.')
    if 'customer_id' in df and df.customer_id.dropna().duplicated().any():
        raise ValueError('Use one row per customer. Repeated customer IDs could leak across the test split.')
    features = [c for c in FEATURES if c in df and df[c].notna().any()]
    if len(features) < 2:
        raise ValueError('Provide at least two Telco predictors, such as Tenure Months, Contract, and Monthly Charges.')
    df['churn_value'] = y.astype(int)
    return df, features


def metrics(y, probability, threshold=0.5):
    predicted = probability >= threshold
    return {'Accuracy': accuracy_score(y, predicted), 'Precision': precision_score(y, predicted, zero_division=0),
            'Recall': recall_score(y, predicted, zero_division=0), 'F1': f1_score(y, predicted, zero_division=0),
            'ROC AUC': roc_auc_score(y, probability)}


def train_models(df, features):
    x, y = df[features], df.churn_value
    development, test = train_test_split(df.index, test_size=0.2, stratify=y, random_state=42)
    train, validation = train_test_split(development, test_size=0.25, stratify=y.loc[development], random_state=42)
    numeric = [c for c in features if c in NUMERIC]
    categorical = [c for c in features if c not in numeric]
    preprocess = ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)), ('scale', StandardScaler())]), numeric),
        ('category', Pipeline([('impute', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
                               ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical)])
    models = {'Logistic regression': LogisticRegression(max_iter=1500, random_state=42),
              'Decision tree': DecisionTreeClassifier(max_depth=6, min_samples_leaf=15, random_state=42),
              'Random forest': RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=1),
              'Gradient boosting': GradientBoostingClassifier(n_estimators=100, max_depth=2, random_state=42)}
    rows, fitted = [], {}
    for name, model in models.items():
        pipe = Pipeline([('preprocess', clone(preprocess)), ('model', model)])
        pipe.fit(x.loc[train], y.loc[train])
        rows.append({'Model': name, **metrics(y.loc[validation], pipe.predict_proba(x.loc[validation])[:, 1])})
        fitted[name] = pipe
    comparison = pd.DataFrame(rows).sort_values('ROC AUC', ascending=False).reset_index(drop=True)
    name = comparison.iloc[0]['Model']
    evaluation_model = clone(fitted[name]).fit(x.loc[development], y.loc[development])
    probability = evaluation_model.predict_proba(x.loc[test])[:, 1]
    # Refit only after evaluation; this separate model is used for operational scoring.
    final_model = clone(fitted[name]).fit(x, y)
    estimator = final_model.named_steps['model']
    importance = (np.abs(estimator.coef_[0]) if hasattr(estimator, 'coef_') else estimator.feature_importances_)
    drivers = pd.DataFrame({'Feature': final_model.named_steps['preprocess'].get_feature_names_out(), 'Importance': importance}).sort_values('Importance', ascending=False)
    return {'model': final_model, 'name': name, 'comparison': comparison, 'features': features,
            'test_y': y.loc[test], 'test_probability': probability, 'drivers': drivers,
            'split_sizes': (len(train), len(validation), len(test))}


def score_customers(raw, result, threshold=0.5):
    if not 0 <= threshold <= 1:
        raise ValueError('Threshold must be between 0 and 1.')
    df = clean_data(raw)
    missing = sorted(set(result['features']) - set(df.columns))
    if missing:
        raise ValueError('Missing prediction columns: ' + ', '.join(missing))
    if df.empty:
        raise ValueError('The prediction sheet contains no customers.')
    probability = result['model'].predict_proba(df[result['features']])[:, 1]
    df['churn_probability'] = probability
    df['predicted_churn'] = (probability >= threshold).astype(int)
    df['risk_level'] = np.select([probability >= 0.7, probability >= 0.3], ['High', 'Medium'], default='Low')
    df['suggested_action'] = df.risk_level.map({'High': 'Review service issues and discuss a suitable retention offer',
                                             'Medium': 'Check satisfaction and plan suitability', 'Low': 'Maintain regular customer support'})
    return df.sort_values('churn_probability', ascending=False)
