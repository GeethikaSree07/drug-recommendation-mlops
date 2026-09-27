import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


# Load dataset
df = pd.read_csv("C:/Users/HP/PycharmProjects/Drug-Recommendation-MLOps/Data/drug200.csv")

# Features and target
X = df.drop("Drug", axis=1)
y = df["Drug"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Feature types
numeric_features = ["Age", "Na_to_K"]
categorical_features = ["Sex", "BP", "Cholesterol"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# Random Forest model
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ]
)

# Train
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
weighted_f1 = f1_score(y_test, y_pred, average="weighted")

# MLflow
mlflow.set_experiment("Drug-Recommendation-System")

with mlflow.start_run():
    mlflow.log_param("model", "Random Forest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("weighted_f1", weighted_f1)

    mlflow.sklearn.log_model(
        model,
        name="drug_recommendation_model",
        skops_trusted_types=["sklearn.tree._tree.Tree"],
        registered_model_name="DrugRecommendationModel"
    )

    print("MLflow training completed!")
    print("Accuracy:", accuracy)
    print("Weighted F1:", weighted_f1)