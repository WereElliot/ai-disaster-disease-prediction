import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import os  # For building paths dynamically

# ---------------- Load Dataset ----------------
def load_data(path):
    df = pd.read_csv(path)
    # Create outbreak target variable
    df["outbreak"] = (df["disease_cases"] > 12).astype(int)
    return df

# ---------------- Prepare Features ----------------
def prepare_features(df):
    X = df[[
        "temperature",
        "humidity",
        "rainfall",
        "population_density",
        "flood_event"
    ]]
    y = df["outbreak"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------- Train Baseline Models ----------------
def train_models(X_train, X_test, y_train, y_test):
    # Logistic Regression
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)

    # Random Forest
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))
    print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))

# ---------------- Main Execution ----------------
if __name__ == "__main__":
    # Automatically find project root folder (one level above src)
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(PROJECT_ROOT, "data", "sample_disease_climate_data.csv")

    # Load dataset
    df = load_data(data_path)

    # Prepare train/test sets
    X_train, X_test, y_train, y_test = prepare_features(df)

    # Train and evaluate models
    train_models(X_train, X_test, y_train, y_test)