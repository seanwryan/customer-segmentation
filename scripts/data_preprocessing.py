# data_preprocessing.py

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Load the datasets
def load_data(train_path="data/Train.csv", test_path="data/Test.csv"):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data

# Preprocess the datasets
def preprocess_data(train_data, test_data):
    # Separate target variable from training data
    target_column = "Segmentation"
    X_train = train_data.drop(columns=["ID", target_column])  # Drop ID column
    y_train = train_data[target_column]
    X_test = test_data.drop(columns=["ID"])  # Drop ID column
    
    # Identify numerical and categorical columns
    numeric_features = ["Age", "Work_Experience", "Family_Size"]
    categorical_features = ["Gender", "Ever_Married", "Graduated", "Profession", "Spending_Score", "Var_1"]

    # Define preprocessing for numerical data: Impute missing values and scale
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    # Define preprocessing for categorical data: Impute missing values and one-hot encode
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine preprocessors in a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # Preprocess the datasets
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    # Encode target variable
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    return X_train_preprocessed, y_train_encoded, X_test_preprocessed, label_encoder

# Save preprocessed data for modeling
def save_preprocessed_data(X_train, y_train, X_test, label_encoder, output_dir="data/processed/"):
    import os
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(X_train).to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    pd.DataFrame(y_train).to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    pd.DataFrame(X_test).to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    # Save label encoder classes
    pd.Series(label_encoder.classes_).to_csv(os.path.join(output_dir, "label_classes.csv"), index=False)

# Main script execution
if __name__ == "__main__":
    # Load data
    train_data, test_data = load_data()

    # Preprocess data
    X_train, y_train, X_test, label_encoder = preprocess_data(train_data, test_data)

    # Save preprocessed data
    save_preprocessed_data(X_train, y_train, X_test, label_encoder)

    print("Data preprocessing completed and saved!")