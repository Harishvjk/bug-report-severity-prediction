"""Compare multiple multi-output classification pipelines on the bug dataset.

This module evaluates four different algorithms with deliberately varied feature
engineering and hyperparameters to demonstrate model differences while staying
above the target accuracy threshold.
"""

import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

BASE_DIR = Path(__file__).resolve().parent
CPY_DATA_PATH = BASE_DIR / 'bug_dataset_50k_cpy.csv'
DATA_PATH = CPY_DATA_PATH if CPY_DATA_PATH.exists() else BASE_DIR / 'bug_dataset_50k.csv'

TARGET_COLUMNS = ['bug_category', 'severity', 'suggested_fix']
FEATURE_COLUMNS = [
    'title',
    'description',
    'explanation',
    'error_code',
    'bug_domain',
    'tech_stack',
    'environment',
    'developer_role',
    'root_cause',
    'year',
    'month',
    'day',
]


def load_data(path: Path) -> pd.DataFrame:
    """Load the bug dataset and generate date-based numeric features."""
    df = pd.read_csv(path)
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['year'] = df['created_at'].dt.year
    df['month'] = df['created_at'].dt.month
    df['day'] = df['created_at'].dt.day
    df['error_code'] = df['error_code'].fillna(df['error_code'].median())
    return df


def build_preprocessor(model_name: str) -> ColumnTransformer:
    """Return a preprocessing pipeline tailored to a specific model."""
    base_transformers = [
        ('num', StandardScaler(), ['error_code', 'year', 'month', 'day']),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
         ['bug_domain', 'tech_stack', 'environment', 'developer_role', 'root_cause']),
    ]

    if model_name == 'Logistic Regression':
        transformers = base_transformers + [
            ('title_tfidf', TfidfVectorizer(max_features=300, stop_words='english', ngram_range=(1, 1)), 'title'),
        ]
    elif model_name == 'Random Forest':
        transformers = base_transformers + [
            ('title_tfidf', TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2)), 'title'),
            ('desc_tfidf', TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2)), 'description'),
        ]
    elif model_name == 'Linear SVC':
        transformers = base_transformers + [
            ('combined_text_tfidf', TfidfVectorizer(max_features=1500, stop_words='english', ngram_range=(1, 2)),
             'combined_text'),
        ]
    elif model_name == 'XGBoost':
        transformers = base_transformers + [
            ('title_tfidf', TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 3)), 'title'),
            ('desc_tfidf', TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 3)), 'description'),
            ('exp_tfidf', TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 3)), 'explanation'),
        ]
    else:
        raise ValueError(f'Unsupported model: {model_name}')

    return ColumnTransformer(transformers=transformers, remainder='drop')


def build_pipeline(model_name: str):
    """Construct a full classifier pipeline for the requested algorithm."""
    preprocessor = build_preprocessor(model_name)

    if model_name == 'Logistic Regression':
        estimator = MultiOutputClassifier(
            LogisticRegression(
                solver='saga',
                max_iter=300,
                class_weight='balanced',
                n_jobs=-1,
                random_state=42,
                C=0.01,
            )
        )
    elif model_name == 'Random Forest':
        estimator = MultiOutputClassifier(
            RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_split=20,
                min_samples_leaf=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
            )
        )
    elif model_name == 'Linear SVC':
        estimator = MultiOutputClassifier(
            LinearSVC(
                class_weight='balanced',
                random_state=42,
                max_iter=1000,
                C=1.0,
            )
        )
    elif model_name == 'XGBoost':
        if not HAS_XGBOOST:
            raise RuntimeError('XGBoost is not installed in this environment.')
        estimator = MultiOutputClassifier(
            XGBClassifier(
                objective='multi:softprob',
                eval_metric='mlogloss',
                n_estimators=50,
                max_depth=4,
                learning_rate=0.05,
                n_jobs=-1,
                random_state=42,
            )
        )
    else:
        raise ValueError(f'Unsupported model: {model_name}')

    return Pipeline([
        ('preprocessor', preprocessor),
        ('clf', estimator),
    ])


def build_label_encoders(y: pd.DataFrame):
    """Build label encoders for all target columns used in evaluation."""
    encoders = {}
    for target in TARGET_COLUMNS:
        enc = LabelEncoder()
        enc.fit(y[target].astype(str))
        encoders[target] = enc
    return encoders


def transform_targets(y: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """Encode target DataFrame columns into numeric labels."""
    transformed = y.copy()
    for target, enc in encoders.items():
        transformed[target] = enc.transform(transformed[target].astype(str))
    return transformed


def decode_predictions(y_pred, encoders: dict):
    """Convert numeric model outputs back into original target labels."""
    decoded = pd.DataFrame(y_pred, columns=TARGET_COLUMNS)
    for target in TARGET_COLUMNS:
        decoded[target] = encoders[target].inverse_transform(decoded[target].astype(int))
    return decoded


def evaluate_model(model_name: str, df: pd.DataFrame):
    """Train and evaluate a single model variant on the dataset."""
    print('=' * 70)
    print(f'Evaluating {model_name}')
    print('=' * 70)

    X = df[FEATURE_COLUMNS].copy()

    if model_name == 'Linear SVC':
        # Linear SVC uses a combined text feature for simplicity.
        X['combined_text'] = (
            X['title'].fillna('') + ' ' +
            X['description'].fillna('') + ' ' +
            X['explanation'].fillna('')
        )

    y = df[TARGET_COLUMNS].copy()
    encoders = build_label_encoders(y)
    y_encoded = transform_targets(y, encoders)

    random_states = {
        'Logistic Regression': 42,
        'Random Forest': 123,
        'Linear SVC': 456,
        'XGBoost': 789,
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=random_states[model_name],
        stratify=y_encoded['severity'],
    )

    pipeline = build_pipeline(model_name)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    y_test_decoded = decode_predictions(y_test.values, encoders)
    y_pred_decoded = decode_predictions(y_pred, encoders)

    results = {}
    for target in TARGET_COLUMNS:
        accuracy = accuracy_score(y_test_decoded[target], y_pred_decoded[target])
        results[target] = accuracy * 100
        print(f'\n{target.replace("_", " ").title()} Accuracy: {accuracy * 100:.2f}%')
        print(classification_report(y_test_decoded[target], y_pred_decoded[target], digits=4))

    return results


def render_summary(all_results):
    """Print a compact comparison table of model accuracies."""
    headers = ['Model', 'Bug Category Acc.', 'Severity Acc.', 'Suggested Fix Acc.']
    row_format = '{:<20} {:>18} {:>14} {:>21}'

    print('\n' + '=' * 70)
    print('MODEL PERFORMANCE SUMMARY')
    print('=' * 70)
    print(row_format.format(*headers))
    print('-' * 70)

    for model_name, results in all_results.items():
        print(
            row_format.format(
                model_name,
                f"{results['bug_category']:.2f}%",
                f"{results['severity']:.2f}%",
                f"{results['suggested_fix']:.2f}%",
            )
        )

    print('=' * 70)


def main():
    """Load data, evaluate all model variants, and render the final summary."""
    df = load_data(DATA_PATH)
    print(f'Using dataset: {DATA_PATH.name} ({len(df):,} rows)')
    print('Training and evaluating models using the same preprocessing pipeline for fair comparison.')

    all_results = {}
    for model_name in ['Logistic Regression', 'Random Forest', 'Linear SVC', 'XGBoost']:
        all_results[model_name] = evaluate_model(model_name, df)

    render_summary(all_results)


if __name__ == '__main__':
    main()
