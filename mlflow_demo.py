import mlflow

# Use SQLite database instead of file store
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# rest stays exactly the same
mlflow.set_experiment("my_first_experiment")

with mlflow.start_run(run_name="run_1"):
    mlflow.log_param("chunk_size", 500)
    mlflow.log_param("k", 3)
    mlflow.log_metric("rouge_score", 0.65)
    mlflow.log_metric("bleu_score", 0.58)

with mlflow.start_run(run_name="run_2"):
    mlflow.log_param("chunk_size", 1000)
    mlflow.log_param("k", 5)
    mlflow.log_metric("rouge_score", 0.78)
    mlflow.log_metric("bleu_score", 0.71)

print("Done! Check http://127.0.0.1:5000")