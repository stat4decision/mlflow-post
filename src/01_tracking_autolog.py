"""
Exemple 1 : le suivi automatique (autolog).

L'autolog est le moyen le plus simple de commencer avec MLflow : une seule ligne,
et MLflow enregistre tout seul les paramètres, les métriques et le modèle.

Pour lancer ce script :
    python src/01_tracking_autolog.py

Par défaut, MLflow écrit dans un dossier local "mlruns/". Pour visualiser les
résultats, lancez ensuite "mlflow ui" dans le même dossier.
"""

import mlflow
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# On donne un nom à notre expérimentation (un dossier logique qui regroupe les essais).
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("classification_vins")

# Une seule ligne suffit : MLflow va tout enregistrer automatiquement.
mlflow.sklearn.autolog()

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chaque "run" est un essai. Le bloc "with" ouvre et ferme proprement l'essai.
with mlflow.start_run(run_name="random_forest_autolog"):
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    print("Modele entraine. Lancez 'mlflow ui' pour voir le resultat.")
