import pickle
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
CPY_DATA_PATH = BASE_DIR / 'bug_dataset_50k_cpy.csv'
DATA_PATH = CPY_DATA_PATH if CPY_DATA_PATH.exists() else BASE_DIR / 'bug_dataset_50k.csv'

# Model paths
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
    """Load and preprocess the dataset."""
    df = pd.read_csv(path)
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['year'] = df['created_at'].dt.year
    df['month'] = df['created_at'].dt.month
    df['day'] = df['created_at'].dt.day
    return df


def load_multi_model():
    """Load the multi-output model and its components."""
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
    """Prepare features for multi-output model evaluation."""
    X = df[['title', 'description', 'explanation', 'error_code', 'bug_domain', 'tech_stack',
            'environment', 'developer_role', 'root_cause', 'year', 'month', 'day']].copy()
    y = df[['bug_category', 'severity', 'suggested_fix']].copy()

    # Encode categorical features
    for col in ['bug_domain', 'tech_stack', 'environment', 'developer_role', 'root_cause']:
        X[col] = encoders[col].transform(X[col].astype(str))

    X['error_code'] = X['error_code'].fillna(X['error_code'].median())

    # Transform targets to encoded labels
    y_encoded = y.copy()
    for col in ['bug_category', 'severity', 'suggested_fix']:
        y_encoded[col] = encoders[col].transform(y[col].astype(str))

    return X, y_encoded, y


def evaluate_multi_model(df: pd.DataFrame):
    """Evaluate the multi-output model."""
    print("=" * 60)
    print("MULTI-OUTPUT MODEL EVALUATION")
    print("=" * 60)

    try:
        model, preprocessor, encoders = load_multi_model()
    except FileNotFoundError as e:
        print(f"Error loading multi-output model: {e}")
        return

    X, y_encoded, y_original = prepare_features_for_multi(df, encoders)

    # Split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded['severity']
    )

    # Transform features
    X_test_transformed = preprocessor.transform(X_test)
    y_pred_encoded = model.predict(X_test_transformed)

    # Convert predictions back to original labels
    targets = ['bug_category', 'severity', 'suggested_fix']
    for i, target in enumerate(targets):
        y_test_original = encoders[target].inverse_transform(y_test[target])
        y_pred_original = encoders[target].inverse_transform(y_pred_encoded[:, i])

        print(f"\n{target.upper()} PREDICTIONS:")
        print("-" * 40)
        print(f"Accuracy: {accuracy_score(y_test_original, y_pred_original):.4f}")

        # Detailed classification report
        print("\nClassification Report:")
        print(classification_report(y_test_original, y_pred_original, digits=4))

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test_original, y_pred_original, average=None
        )
        print("Per-class metrics:")
        classes = encoders[target].classes_
        for cls, p, r, f, s in zip(classes, precision, recall, f1, support):
            print(f"  {cls}: Precision={p:.4f}, Recall={r:.4f}, F1={f:.4f}, Support={s}")


def build_severity_pipeline():
    """Build the severity-only pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2)),
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
    """Train and evaluate the severity-only model."""
    print("\n" + "=" * 60)
    print("SEVERITY-ONLY MODEL EVALUATION")
    print("=" * 60)

    # Prepare data for severity model
    df['combined_text'] = (df['title'].fillna('') + ' ' +
                          df['description'].fillna('') + ' ' +
                          df['explanation'].fillna(''))

    X = df[['combined_text', 'error_code', 'year', 'month', 'day',
            'bug_domain', 'tech_stack', 'environment', 'developer_role', 'bug_category']]
    y = df['severity']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    clf = build_severity_pipeline()
    clf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = clf.predict(X_test)

    print(f"Severity Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    if accuracy_score(y_test, y_pred) > 0.90:
        print("✓ SUCCESS: Severity accuracy is above 90%")
    else:
        print("⚠ WARNING: Severity accuracy is below 90%")

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None
    )
    print("Per-class metrics:")
    classes = sorted(y_test.unique())
    for cls, p, r, f, s in zip(classes, precision, recall, f1, support):
        print(f"  {cls}: Precision={p:.4f}, Recall={r:.4f}, F1={f:.4f}, Support={s}")


def main():
    """Main evaluation function."""
    print("Bug Prediction Model Accuracy Report")
    print("=" * 60)
    print(f"Dataset: {DATA_PATH}")
    print(f"Dataset size: {len(pd.read_csv(DATA_PATH)):,} records")
    print()

    # Load dataset
    df = load_data(DATA_PATH)

    # Evaluate multi-output model
    evaluate_multi_model(df)

    # Evaluate severity-only model
    evaluate_severity_model(df)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()