# model_training.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib
import os

# Load preprocessed data
def load_preprocessed_data(data_dir="data/processed/"):
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).squeeze()  # Convert to Series
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    label_classes = pd.read_csv(os.path.join(data_dir, "label_classes.csv"), header=None).squeeze().to_list()
    return X_train, y_train, X_test, label_classes

# Train the model
def train_model(X_train, y_train):
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Evaluate the model
def evaluate_model(model, X_train, y_train):
    print("Evaluating model...")
    y_train_pred = model.predict(X_train)

    # Classification metrics
    report = classification_report(y_train, y_train_pred)
    matrix = confusion_matrix(y_train, y_train_pred)
    accuracy = accuracy_score(y_train, y_train_pred)
    f1 = f1_score(y_train, y_train_pred, average="weighted")

    print("\nClassification Report:")
    print(report)

    print("\nConfusion Matrix:")
    print(matrix)

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return accuracy, f1, report, matrix

# Save the model and evaluation metrics
def save_model_and_metrics(model, accuracy, f1, report, matrix, output_dir="models/"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the model
    model_path = os.path.join(output_dir, "random_forest_model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Save evaluation metrics
    metrics_path = os.path.join(output_dir, "evaluation_report.txt")
    with open(metrics_path, "w") as f:
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write("Confusion Matrix:\n")
        f.write(str(matrix) + "\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
    print(f"Evaluation metrics saved to {metrics_path}")

# Main script execution
if __name__ == "__main__":
    # Load data
    X_train, y_train, X_test, label_classes = load_preprocessed_data()

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    accuracy, f1, report, matrix = evaluate_model(model, X_train, y_train)

    # Save model and metrics
    save_model_and_metrics(model, accuracy, f1, report, matrix)

    # Summary of results
    if accuracy > 0.8 and f1 > 0.8:
        print("\nThe model is performing well. Consider publishing the project!")
    else:
        print("\nThe model needs further improvement before publishing.")