"""
Point d'entree du MLflow Project (voir le fichier MLproject).

Ce script permet de relancer un entrainement de facon reproductible, avec des
parametres passes en ligne de commande.

Exemples :
    python train.py --n-estimators 200 --max-depth 15
    mlflow run . -P n_estimators=200 -P max_depth=15
"""

import argparse

import mlflow
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    X, y = load_wine(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )

    with mlflow.start_run():
        mlflow.log_params(vars(args))
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42,
        )
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, name="model")
        print(f"Accuracy : {accuracy:.4f}")


if __name__ == "__main__":
    main()
