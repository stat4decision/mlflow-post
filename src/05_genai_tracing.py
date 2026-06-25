"""
Exemple 5 : le suivi (tracing) pour l'IA generative, nouveaute de MLflow 3.

MLflow 3 sait tracer les appels aux LLM et aux agents : entrees, sorties, etapes,
latence. Cela aide a comprendre et a deboguer une application a base de LLM.

Cet exemple est volontairement minimal et ne necessite pas de cle d'API : on
instrumente une simple fonction avec le decorateur @mlflow.trace.

Pour lancer ce script :
    python src/05_genai_tracing.py
"""

import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("demo_genai")

# Pour tracer automatiquement les vrais appels a un fournisseur, une ligne suffit :
#     mlflow.openai.autolog()
#     mlflow.langchain.autolog()


# Le decorateur @mlflow.trace enregistre les entrees, sorties et la duree.
@mlflow.trace
def repondre(question: str) -> str:
    """Simulation d'une reponse de LLM (a remplacer par votre vraie logique)."""
    base = {
        "Qu'est-ce que MLflow ?": "Une plateforme open source de gestion du cycle de vie ML.",
    }
    return base.get(question, "Je ne sais pas encore repondre a cette question.")


if __name__ == "__main__":
    print(repondre("Qu'est-ce que MLflow ?"))
    print("Trace enregistree. Ouvrez 'mlflow ui' puis l'onglet Traces.")
