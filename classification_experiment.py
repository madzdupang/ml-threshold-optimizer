from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from threshold_optimizer import find_best_threshold
from sklearn.ensemble import RandomForestClassifier


def split_dataset(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.40,
        random_state=42,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_thresholds(probabilities, y_val, thresholds, minimum_recall):
    precisions = []
    recalls = []
    for threshold in thresholds:
        predictions = np.where(probabilities >= threshold, 0, 1)

        precision = precision_score(
            y_val, predictions, pos_label=0, zero_division=0)
        recall = recall_score(y_val, predictions,
                              pos_label=0, zero_division=0)

        precisions.append(precision)
        recalls.append(recall)

    best_threshold_result = find_best_threshold(
        thresholds, precisions, recalls, minimum_recall)

    if best_threshold_result is not None:
        best_threshold, best_precision = best_threshold_result
        idx = next(i for i, t in enumerate(thresholds) if t == best_threshold)
        return best_threshold, best_precision, recalls[idx]
    else:
        return None


def main():
    # Your existing ML experiment code
    data = load_breast_cancer()

    X = data.data
    y = data.target
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

    print("Features shape:", X.shape)
    print("Labels shape:", y.shape)
    print("Classes:", data.target_names)

    print("Training:", len(X_train))
    print("Validation:", len(X_val))
    print("Testing:", len(X_test))

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42)
    )

    random_forest = RandomForestClassifier(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)

    random_forest.fit(X_train, y_train)
    print("Model training completed!")

    probabilities = model.predict_proba(X_val)
    rf_probabilities = random_forest.predict_proba(X_val)

    print("Model classes:", model.classes_)
    print("Probability array shape:", probabilities.shape)
    print("First 5 probability pairs:")
    print(probabilities[:5,])

    malignant_probabilities = probabilities[:, 0]
    rf_malignant_probabilities = rf_probabilities[:, 0]

    thresholds = np.arange(0.1, 1.0, 0.1)
    minimum_recall = 0.95

    rf_result = evaluate_thresholds(
        rf_malignant_probabilities, y_val, thresholds, minimum_recall)

    lr_result = evaluate_thresholds(
        malignant_probabilities, y_val, thresholds, minimum_recall)

    print("Random Forest Result:", rf_result)
    print("Logistic Regression Result:", lr_result)


if __name__ == "__main__":
    main()
