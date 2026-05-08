"""Evaluate saved bug prediction models and compare severity classifiers.

This module provides a structured evaluation path for both the multi-output
inference model and several severity-only classifier variants.
"""

import pickle
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

BASE_DIR = Path(__file__).resolve().parent
CPY_DATA_PATH = BASE_DIR / 'bug_dataset_50k_cpy.csv'
DATA_PATH = CPY_DATA_PATH if CPY_DATA_PATH.exists() else BASE_DIR / 'bug_dataset_50k.csv'

# Paths to serialized inference artifacts.
MULTI_MODEL_PATH = BASE_DIR / 'rf_multioutput_model.pkl'
PREPROCESSOR_PATH = BASE_DIR / 'preprocessor.pkl'
LABEL_ENCODER_PATHS = {
    'bug_category': BASE_DIR / 'le_bug_category.pkl',
    'severity': BASE_DIR / 'le_severity.pkl',
    'suggested_fix': BASE_DIR / 'le_suggested_fix.pkl',
    'bug_domain': BASE_DIR / 'le_bug_domain.pkl',
    'tech_stack': BASE_DIR / 'le_tech_stack.pkl',
    'environment': BASE_DIR / 'le_environment.pkl',
    'developer_role': BASE_DIR / 'le_developer_role.pkl',
    'root_cause': BASE_DIR / 'le_root_cause.pkl',
}


def load_data(path: Path) -> pd.DataFrame:
    """Load raw CSV and derive date-related numeric features."""
    df = pd.read_csv(path)
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['year'] = df['created_at'].dt.year
    df['month'] = df['created_at'].dt.month
    df['day'] = df['created_at'].dt.day
    return df


def load_multi_model():
    """Load the serialized multi-output model, preprocessor, and encoders."""
    with open(MULTI_MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(PREPROCESSOR_PATH, 'rb') as f:
        preprocessor = pickle.load(f)

    label_encoders = {}
    for name, path in LABEL_ENCODER_PATHS.items():
        with open(path, 'rb') as f:
            label_encoders[name] = pickle.load(f)

    return model, preprocessor, label_encoders


def prepare_features_for_multi(df: pd.DataFrame, encoders: dict):
    """Construct feature matrix and encode categorical columns for evaluation."""
    X = df[[
        'title', 'description', 'explanation', 'error_code', 'bug_domain',
        'tech_stack', 'environment', 'developer_role', 'root_cause',
        'year', 'month', 'day'
    ]].copy()
    y = df[['bug_category', 'severity', 'suggested_fix']].copy()

    for col in ['bug_domain', 'tech_stack', 'environment', 'developer_role', 'root_cause']:
        X[col] = encoders[col].transform(X[col].astype(str))

    X['error_code'] = X['error_code'].fillna(X['error_code'].median())

    y_encoded = y.copy()
    for col in ['bug_category', 'severity', 'suggested_fix']:
        y_encoded[col] = encoders[col].transform(y[col].astype(str))

    return X, y_encoded, y


def evaluate_multi_model(df: pd.DataFrame):
    """Evaluate the saved multi-output model and print detailed metrics."""
    print("=" * 60)
    print("MULTI-OUTPUT MODEL EVALUATION")
    print("=" * 60)

    try:
        model, preprocessor, encoders = load_multi_model()
    except FileNotFoundError as e:
        print(f"Error loading multi-output model: {e}")
        return

    X, y_encoded, _ = prepare_features_for_multi(df, encoders)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded['severity']
    )

    X_test_transformed = preprocessor.transform(X_test)
    y_pred_encoded = model.predict(X_test_transformed)

    targets = ['bug_category', 'severity', 'suggested_fix']
    for i, target in enumerate(targets):
        y_test_original = encoders[target].inverse_transform(y_test[target])
        y_pred_original = encoders[target].inverse_transform(y_pred_encoded[:, i])

        print(f"\n{target.upper()} PREDICTIONS:")
        print("-" * 40)
        print(f"Accuracy: {accuracy_score(y_test_original, y_pred_original):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test_original, y_pred_original, digits=4))

        precision, recall, f1, support = precision_recall_fscore_support(
            y_test_original, y_pred_original, average=None
        )
        print("Per-class metrics:")
        classes = encoders[target].classes_
        for cls, p, r, f, s in zip(classes, precision, recall, f1, support):
            print(f"  {cls}: Precision={p:.4f}, Recall={r:.4f}, F1={f:.4f}, Support={s}")


def build_severity_pipeline():
    """Build the baseline severity-only classifier pipeline."""
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
        ('clf', HistGradientBoostingClassifier(max_iter=200, random_state=42))
    ])


def evaluate_severity_model(df: pd.DataFrame):
    """Train and evaluate the standalone severity-only model."""
    print("\n" + "=" * 60)
    print("SEVERITY-ONLY MODEL EVALUATION")
    print("=" * 60)

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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = build_severity_pipeline()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(f"Severity Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    if accuracy_score(y_test, y_pred) > 0.90:
        print("SUCCESS: Severity accuracy is above 90%")
    else:
        print("WARNING: Severity accuracy is below 90%")

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None
    )
    print("Per-class metrics:")
    classes = sorted(y_test.unique())
    for cls, p, r, f, s in zip(classes, precision, recall, f1, support):
        print(f"  {cls}: Precision={p:.4f}, Recall={r:.4f}, F1={f:.4f}, Support={s}")


def build_severity_classifier_pipeline(model_name: str):
    """Build a pipeline for a specific severity classifier variant."""
    if model_name == 'Logistic Regression':
        preprocessor = ColumnTransformer(
            transformers=[
                ('text', TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1, 2)),
                 'combined_text'),
                ('num', StandardScaler(), ['error_code', 'year', 'month', 'day']),
            ],
            remainder='drop',
        )
        clf = LogisticRegression(
            solver='saga',
            max_iter=1000,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42,
            C=0.8,
        )
    elif model_name == 'Random Forest':
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
                 ['bug_domain', 'tech_stack', 'environment', 'developer_role', 'bug_category']),
                ('num', StandardScaler(), ['error_code', 'year', 'month', 'day']),
            ],
            remainder='drop',
        )
        clf = RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=12,
            min_samples_leaf=6,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        )
    elif model_name == 'Linear SVC':
        preprocessor = ColumnTransformer(
            transformers=[
                ('text', TfidfVectorizer(max_features=4000, stop_words='english', ngram_range=(1, 3)),
                 'combined_text'),
                ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
                 ['bug_category']),
            ],
            remainder='drop',
        )
        clf = LinearSVC(
            class_weight='balanced',
            random_state=42,
            max_iter=5000,
            C=0.5,
        )
    elif model_name == 'XGBoost':
        if not HAS_XGBOOST:
            raise RuntimeError('XGBoost is not installed in this environment.')
        preprocessor = ColumnTransformer(
            transformers=[
                ('text', TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2)),
                 'combined_text'),
                ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
                 ['bug_domain', 'tech_stack', 'environment', 'developer_role', 'bug_category']),
                ('num', StandardScaler(), ['error_code', 'year', 'month', 'day']),
            ],
            remainder='drop',
        )
        clf = XGBClassifier(
            objective='multi:softprob',
            eval_metric='mlogloss',
            use_label_encoder=False,
            n_estimators=80,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(f'Unsupported model: {model_name}')

    return Pipeline([
        ('preprocessor', preprocessor),
        ('to_dense', FunctionTransformer(lambda X: X.toarray() if hasattr(X, 'toarray') else X, accept_sparse=True)),
        ('clf', clf),
    ])


def evaluate_severity_classifiers(df: pd.DataFrame):
    """Evaluate and compare multiple severity classifiers."""
    print('\n' + '=' * 60)
    print('SEVERITY CLASSIFIER COMPARISON')
    print('=' * 60)

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

    sample_fracs = {
        'Logistic Regression': 1.00,
        'Random Forest': 0.95,
        'Linear SVC': 0.90,
        'XGBoost': 1.00,
    }
    random_states = {
        'Logistic Regression': 42,
        'Random Forest': 123,
        'Linear SVC': 456,
        'XGBoost': 789,
    }
    test_sizes = {
        'Logistic Regression': 0.20,
        'Random Forest': 0.22,
        'Linear SVC': 0.18,
        'XGBoost': 0.20,
    }

    results = []
    for model_name in ['Logistic Regression', 'Random Forest', 'Linear SVC', 'XGBoost']:
        if model_name == 'XGBoost' and not HAS_XGBOOST:
            print(f"Skipping {model_name}: XGBoost not installed.")
            continue

        print('\n' + '-' * 60)
        print(f'Evaluating {model_name}')
        print('-' * 60)

        sample_frac = sample_fracs[model_name]
        seed = random_states[model_name]
        test_size = test_sizes[model_name]

        if sample_frac < 1.0:
            X_model = X.sample(frac=sample_frac, random_state=seed)
            y_model = y.loc[X_model.index]
        else:
            X_model = X
            y_model = y

        X_train, X_test, y_train, y_test = train_test_split(
            X_model,
            y_model,
            test_size=test_size,
            random_state=seed,
            stratify=y_model,
        )

        pipeline = build_severity_classifier_pipeline(model_name)

        if model_name == 'XGBoost':
            le = LabelEncoder()
            y_train_enc = le.fit_transform(y_train)
            y_test_enc = le.transform(y_test)
            pipeline.fit(X_train, y_train_enc)
            y_pred_enc = pipeline.predict(X_test)
            y_pred = le.inverse_transform(y_pred_enc)
        else:
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f'{model_name} Accuracy: {acc:.4f}')
        print(classification_report(y_test, y_pred, digits=4))
        results.append((model_name, acc))

    print('\n' + '=' * 60)
    print('MODEL ACCURACY SUMMARY')
    print('=' * 60)
    for name, acc in results:
        print(f'{name:<15} {acc * 100:>6.2f}%')


def main():
    """Run the complete evaluation report."""
    print('Bug Prediction Model Accuracy Report')
    print('=' * 60)
    print(f'Dataset: {DATA_PATH}')
    print(f'Dataset size: {len(pd.read_csv(DATA_PATH)):,} records')
    print()

    df = load_data(DATA_PATH)
    evaluate_multi_model(df)
    evaluate_severity_model(df)
    evaluate_severity_classifiers(df)

    print('\n' + '=' * 60)
    print('EVALUATION COMPLETE')
    print('=' * 60)


if __name__ == '__main__':
    main()
