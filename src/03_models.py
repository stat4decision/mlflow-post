"""
Exemple 3 : packager et recharger un modele, et creer un modele personnalise.

MLflow Models enregistre un modele dans un format standard, ce qui permet de le
recharger plus tard ou de le servir, quel que soit l'outil qui l'a produit.

Pour lancer ce script :
    python src/03_models.py
"""

import mlflow
import mlflow.pyfunc
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("classification_vins")

wine = load_wine()
X, y = wine.data, wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Enregistrer un modele simple, puis le recharger ---
model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)

with mlflow.start_run(run_name="enregistrement_modele"):
    mlflow.sklearn.log_model(
        sk_model=model,
        name="wine_classifier",
        registered_model_name="WineClassifier",
        # La signature decrit les entrees et sorties attendues : elle fiabilise le service.
        signature=mlflow.models.infer_signature(X_train, model.predict(X_train)),
    )

loaded = mlflow.sklearn.load_model("models:/WineClassifier/1")
print("Predictions du modele recharge :", loaded.predict(X_test)[:5])


# --- Un modele personnalise avec PyFunc (preprocessing inclus) ---
class WineClassifierWrapper(mlflow.pyfunc.PythonModel):
    """Enveloppe qui applique un preprocessing avant la prediction."""

    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    # Depuis MLflow 2.6, la signature inclut params=None.
    def predict(self, context, model_input, params=None):
        X = self.scaler.transform(model_input)
        preds = self.model.predict(X)
        confidence = self.model.predict_proba(X).max(axis=1)
        return pd.DataFrame({
            "prediction": preds,
            "confidence": confidence,
            "wine_type": [wine.target_names[p] for p in preds],
        })


scaler = StandardScaler().fit(X_train)
model_scaled = RandomForestClassifier(n_estimators=100, random_state=42).fit(
    scaler.transform(X_train), y_train
)

with mlflow.start_run(run_name="modele_personnalise"):
    mlflow.pyfunc.log_model(
        name="custom_wine_classifier",
        python_model=WineClassifierWrapper(model_scaled, scaler),
        registered_model_name="CustomWineClassifier",
    )
print("Modele personnalise enregistre.")
