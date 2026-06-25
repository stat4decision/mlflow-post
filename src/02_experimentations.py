"""
Exemple 2 : le suivi manuel, pour tracer exactement ce que l'on veut.

L'autolog est pratique, mais dès que l'on veut comparer plusieurs configurations
ou enregistrer des metriques personnalisees, on utilise le logging explicite.

Pour lancer ce script :
    python src/02_experimentations.py
"""

import mlflow
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("classification_vins")

wine = load_wine()
X, y = wine.data, wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(model_type, **params):
    """Entraine un modele et enregistre tout dans MLflow."""
    with mlflow.start_run(run_name=f"{model_type}"):
        # 1. On enregistre les parametres de l'essai.
        mlflow.log_params(params)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("dataset_size", len(X_train))

        # 2. On entraine le modele choisi.
        if model_type == "RandomForest":
            model = RandomForestClassifier(**params, random_state=42)
        elif model_type == "LogisticRegression":
            model = LogisticRegression(**params, random_state=42)
        model.fit(X_train, y_train)

        # 3. On calcule et on enregistre les metriques.
        y_pred = model.predict(X_test)
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred, average="weighted"))
        mlflow.log_metric("recall", recall_score(y_test, y_pred, average="weighted"))

        # 4. On enregistre le modele lui-meme (parametre "name" depuis MLflow 3).
        mlflow.sklearn.log_model(model, name="model")

        # 5. On enregistre un artefact : l'importance des variables.
        if hasattr(model, "feature_importances_"):
            importance = pd.DataFrame({
                "feature": wine.feature_names,
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=False)
            importance.to_csv("feature_importance.csv", index=False)
            mlflow.log_artifact("feature_importance.csv")

        print(f"{model_type:>20} -> accuracy {accuracy_score(y_test, y_pred):.4f}")


if __name__ == "__main__":
    configs = [
        ("RandomForest", {"n_estimators": 100, "max_depth": 10}),
        ("RandomForest", {"n_estimators": 200, "max_depth": 15}),
        ("LogisticRegression", {"C": 1.0, "solver": "lbfgs", "max_iter": 1000}),
        ("LogisticRegression", {"C": 0.1, "solver": "lbfgs", "max_iter": 1000}),
    ]
    for model_type, params in configs:
        train_model(model_type, **params)
