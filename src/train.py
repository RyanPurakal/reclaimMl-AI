from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

FEATURES = ["rolling_volume_7d", "rolling_volume_14d", "sleep_avg_7d", "sleep_std_7d", "days_since_rest", "acwr"]
TARGET = "recovery_risk"

def train_model(df):
    df = df.dropna().reset_index(drop=True)  # remove initial NaNs

    X = df[FEATURES]
    y = df[TARGET]

    # simple train/test split (70/30)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))
    ])

    model.fit(X_train, y_train)

    # Return model + test data for evaluation
    return model, X_test, y_test

def print_feature_importance(model, feature_names):
    # Only works for the final estimator in pipeline
    clf = model.named_steps["clf"]
    coefs = clf.coef_[0]

    print("\nFeature Importance:")
    for f, c in zip(feature_names, coefs):
        print(f"{f}: {c:.3f}")
