import pickle
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
CPY_DATA_PATH = BASE_DIR / 'bug_dataset_50k_cpy.csv'
DATA_PATH = CPY_DATA_PATH if CPY_DATA_PATH.exists() else BASE_DIR / 'bug_dataset_50k.csv'
MODEL_PATH = BASE_DIR / 'rf_multioutput_model.pkl'
PREPROCESSOR_PATH = BASE_DIR / 'preprocessor.pkl'

LABEL_ENCODERS = {
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
    df = pd.read_csv(path)
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['year'] = df['created_at'].dt.year
    df['month'] = df['created_at'].dt.month
    df['day'] = df['created_at'].dt.day
    return df


def build_label_encoders(df: pd.DataFrame):
    encoders = {}
    for name in ['bug_category', 'severity', 'suggested_fix']:
        enc = LabelEncoder()
        enc.fit(df[name].astype(str))
        encoders[name] = enc
    for name in ['bug_domain', 'tech_stack', 'environment', 'developer_role', 'root_cause']:
        enc = LabelEncoder()
        enc.fit(df[name].astype(str))
        encoders[name] = enc
    return encoders


def prepare_features(df: pd.DataFrame, encoders: dict):
    X = df[['title', 'description', 'explanation', 'error_code', 'bug_domain', 'tech_stack', 'environment', 'developer_role', 'root_cause', 'year', 'month', 'day']].copy()
    y = df[['bug_category', 'severity', 'suggested_fix']].copy()

    # Encode target labels and categorical feature values
    y['bug_category'] = encoders['bug_category'].transform(y['bug_category'].astype(str))
    y['severity'] = encoders['severity'].transform(y['severity'].astype(str))
    y['suggested_fix'] = encoders['suggested_fix'].transform(y['suggested_fix'].astype(str))

    X['bug_domain'] = encoders['bug_domain'].transform(X['bug_domain'].astype(str))
    X['tech_stack'] = encoders['tech_stack'].transform(X['tech_stack'].astype(str))
    X['environment'] = encoders['environment'].transform(X['environment'].astype(str))
    X['developer_role'] = encoders['developer_role'].transform(X['developer_role'].astype(str))
    X['root_cause'] = encoders['root_cause'].transform(X['root_cause'].astype(str))

    X['error_code'] = X['error_code'].fillna(X['error_code'].median())

    return X, y


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ('title_tfidf', TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 2)), 'title'),
            ('desc_tfidf', TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 2)), 'description'),
            ('exp_tfidf', TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 2)), 'explanation'),
            ('num_features', StandardScaler(), ['error_code', 'year', 'month', 'day']),
            ('cat_features', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ['tech_stack', 'environment', 'bug_domain', 'developer_role', 'root_cause']),
        ],
        remainder='drop',
    )


def train_model(X, y, preprocessor):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    model = MultiOutputClassifier(
        RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=15,
            min_samples_leaf=8,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        )
    )
    model.fit(X_train_transformed, y_train)

    return model, preprocessor, X_test_transformed, y_test


def save_artifacts(model, preprocessor, encoders):
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(PREPROCESSOR_PATH, 'wb') as f:
        pickle.dump(preprocessor, f)
    for name, path in LABEL_ENCODERS.items():
        with open(path, 'wb') as f:
            pickle.dump(encoders[name], f)


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print('Evaluation on hold-out test set')
    for i, col in enumerate(y_test.columns):
        print(col)
        print('  accuracy:', (y_test.iloc[:, i] == y_pred[:, i]).mean())


def main():
    print(f"Using dataset: {DATA_PATH}")
    df = load_data(DATA_PATH)
    encoders = build_label_encoders(df)
    X, y = prepare_features(df, encoders)
    preprocessor = build_preprocessor()
    model, preprocessor, X_test, y_test = train_model(X, y, preprocessor)
    save_artifacts(model, preprocessor, encoders)
    evaluate(model, X_test, y_test)


if __name__ == '__main__':
    main()
