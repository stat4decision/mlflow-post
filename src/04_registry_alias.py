"""
Exemple 4 : gerer les versions de modeles avec des ALIAS.

Important : les anciens "stages" (Staging, Production) sont depuis MLflow 2.9
remplaces par les ALIAS, plus souples. On peut poser plusieurs alias par version,
et un alias (par exemple "champion") suit la version que l'on veut servir.

Pre-requis : avoir deja enregistre un modele (lancez d'abord src/03_models.py),
ou ce script en enregistre un.

Pour lancer ce script :
    python src/04_registry_alias.py
"""

import mlflow
from mlflow import MlflowClient
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("classification_vins")

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

client = MlflowClient()
model_name = "WineClassificationModel"

# 1. On enregistre une version du modele.
with mlflow.start_run(run_name="registry_demo"):
    model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    mlflow.sklearn.log_model(model, name="model", registered_model_name=model_name)

# 2. On pose un alias "challenger" sur la version 1 (un candidat a tester).
client.set_registered_model_alias(name=model_name, alias="challenger", version=1)

# 3. On annote et on tague la version pour garder une trace.
client.update_model_version(
    name=model_name, version=1,
    description="Classification des vins, RandomForest optimise",
)
client.set_model_version_tag(name=model_name, version=1, key="validation_status", value="passed")

# 4. Apres validation, on promeut la version en "champion" (le modele a servir).
client.set_registered_model_alias(name=model_name, alias="champion", version=1)

# 5. On recharge par alias : la reference suit automatiquement la bonne version.
champion = mlflow.sklearn.load_model(f"models:/{model_name}@champion")
print("Predictions du champion :", champion.predict(X_test)[:5])

# Sur Databricks, le registre vit dans Unity Catalog. On l'active ainsi :
#     mlflow.set_registry_uri("databricks-uc")
# et les modeles se nomment sur trois niveaux : catalogue.schema.modele
