import mlflow
import mlflow.sklearn

mlflow.set_experiment("Drug-Recommendation-System")

with mlflow.start_run():
    mlflow.log_param("model", "Random Forest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)

    mlflow.log_metric("accuracy", 0.975)
    mlflow.log_metric("weighted_f1", 0.97)

    print("MLflow run completed successfully!")