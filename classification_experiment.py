from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from threshold_optimizer import find_best_threshold

data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.40,
    random_state=42,
    stratify=y
)

print("Features shape:", X.shape)
print("Labels shape:", y.shape)
print("Classes:", data.target_names)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=.50,
    random_state=42,
    stratify=y_temp
)

print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(X_test))


model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=42)
)

model.fit(X_train, y_train)

print("Model training completed!")

probabilities = model.predict_proba(X_val)

print("Model classes:", model.classes_)
print("Probability array shape:", probabilities.shape)
print("First 5 probability pairs:")
print(probabilities[:5,])

malignant_probabilities = probabilities[:, 0]

thresholds = np.arange(0.1, 1.0, 0.1)
minimum_recall = 0.95

precisions = []
recalls = []

for threshold in thresholds:
    predictions = np.where(malignant_probabilities >= threshold, 0, 1)

    precision = precision_score(
        y_val, predictions, pos_label=0, zero_division=0)
    recall = recall_score(y_val, predictions, pos_label=0, zero_division=0)

    precisions.append(precision)
    recalls.append(recall)

    if recall >= minimum_recall:
        print(
            f"Threshold: {threshold:.1f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

best_threshold_result = find_best_threshold(
    thresholds, precisions, recalls, minimum_recall)

if best_threshold_result is not None:
    best_threshold, best_precision = best_threshold_result
    idx = np.where(thresholds == best_threshold)[0][0]
    print(
        f"Best Threshold: {best_threshold:.1f} | Best Precision: {best_precision:.4f} | Best Recall: {recalls[idx]:.4f}")
else:
    print("No threshold meets the minimum recall requirement.")

test_probabilities = model.predict_proba(X_test)[:, 0]

test_predictions = np.where(test_probabilities >= best_threshold, 0, 1)

test_precision = precision_score(
    y_test, test_predictions,
    pos_label=0, zero_division=0
)

test_recall = recall_score(
    y_test, test_predictions,
    pos_label=0, zero_division=0
)

print("TEST Precision:", test_precision)
print("TEST Recall:", test_recall)


cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1]
)

print(cm)

print("Test passed!")
