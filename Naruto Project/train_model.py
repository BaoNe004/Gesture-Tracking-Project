import csv
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

CSV_PATH = "landmarks_dataset.csv"
MODEL_PATH = "landmark_rf.joblib"

def load_csv(path):
    X = []
    y = []

    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in reader:
            y.append(row[0])
            features = [float(v) for v in row[1:]]
            X.append(features)

    return np.array(X, dtype=np.float32), np.array(y)

def main():
    X, y = load_csv(CSV_PATH)

    print("samples:", len(X))
    unique, counts = np.unique(y, return_counts=True)
    print("class counts:", dict(zip(unique, counts)))

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)

    print(f"validation accuracy: {acc:.4f}")
    print("\nclassification report:")
    print(classification_report(y_val, y_pred))
    print("confusion matrix:")
    print(confusion_matrix(y_val, y_pred))

    joblib.dump({
        "model": model,
        "class_names": sorted(list(set(y)))
    }, MODEL_PATH)

    print(f"saved model to {MODEL_PATH}")

if __name__ == "__main__":
    main()