"""Train and evaluate a severity-only classifier on the synthetic dataset copy.

This script builds a lightweight classification pipeline for severity prediction
using TF-IDF text features plus numeric and categorical fields.
"""

import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'bug_dataset_50k_cpy.csv'


def load_data(path: Path) -> pd.DataFrame:
    """Load the dataset and create date-derived features for the model."""
    df = pd.read_csv(path)
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['year'] = df['created_at'].dt.year
    df['month'] = df['created_at'].dt.month
    df['day'] = df['created_at'].dt.day
    return df


def build_pipeline() -> Pipeline:
    """Create the preprocessing and classifier pipeline for severity prediction."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2)),
             'combined_text'),
            ('num', StandardScaler(), ['error_code', 'year', 'month', 'day']),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
             ['bug_domain', 'tech_stack', 'environment', 'developer_role', 'bug_category']),
        ],
        remainder='drop',
    )

    return Pipeline([
        ('preprocessor', preprocessor),
        ('to_dense', FunctionTransformer(lambda X: X.toarray() if hasattr(X, 'toarray') else X, accept_sparse=True)),
        ('clf', HistGradientBoostingClassifier(max_iter=200, random_state=42)),
    ])


def main():
    """Load data, train the severity model, and print evaluation results."""
    df = load_data(DATA_PATH)

    # Combine text fields into a single feature for severity modeling.
    df['combined_text'] = (
        df['title'].fillna('') + ' ' +
        df['description'].fillna('') + ' ' +
        df['explanation'].fillna('')
    )

    X = df[[
        'combined_text', 'error_code', 'year', 'month', 'day',
        'bug_domain', 'tech_stack', 'environment', 'developer_role', 'bug_category'
    ]]
    y = df['severity']

    # Use stratified split to preserve severity class balance.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = build_pipeline()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print('Severity accuracy on cpy dataset:', acc)
    print('\nClassification report:')
    print(classification_report(y_test, y_pred, digits=4))

    if acc > 0.90:
        print('\nSUCCESS: severity accuracy is above 90%')
    else:
        print('\nWARNING: severity accuracy is below 90%')


if __name__ == '__main__':
    main()
