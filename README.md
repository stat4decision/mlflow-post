# Guide MLflow par Stat4decision

Ce dépôt rassemble tout le code de l'article de blog **« MLflow : maîtriser le cycle de vie de vos modèles de machine learning »**, publié sur https://www.stat4decision.com/fr/mlflow-modeles-machine-learning/.

L'objectif est simple : vous permettre de tester MLflow vous-même, étape par étape, sur un petit jeu de données (la classification des vins de scikit-learn). Chaque script est court, commenté, et se lance indépendamment.

Le code est à jour pour **MLflow 3**.

## Ce que contient le dépôt

| Fichier | Ce qu'il vous apprend |
|---|---|
| `src/01_tracking_autolog.py` | Le suivi automatique (autolog), la façon la plus simple de démarrer. |
| `src/02_experimentations.py` | Le suivi manuel : enregistrer paramètres, métriques et artefacts. |
| `src/03_models.py` | Enregistrer, recharger et personnaliser un modèle (PyFunc). |
| `src/04_registry_alias.py` | Gérer les versions avec des alias (la méthode actuelle, qui remplace les stages). |
| `src/05_genai_tracing.py` | Le suivi (tracing) pour l'IA générative, nouveauté de MLflow 3. |
| `train.py` + `MLproject` | Un projet MLflow reproductible, lançable en une commande. |

## Prérequis

Vous avez besoin de Python 3.10 ou plus récent. Des bases en Python et en scikit-learn suffisent pour suivre.

## Installation

Ce dépôt utilise un fichier `pyproject.toml` et le gestionnaire [uv](https://docs.astral.sh/uv/), rapide et reproductible.

### Avec uv (recommandé)

```bash
git clone https://github.com/stat4decision/mlflow-guide.git
cd mlflow-guide
uv sync          # crée l'environnement à l'identique depuis uv.lock
```

Vous n'avez pas besoin d'activer l'environnement : préfixez vos commandes par `uv run`.

### Sans uv (avec pip)

Si vous préférez pip, installez les dépendances dans un environnement virtuel :

```bash
python -m venv .venv
source .venv/bin/activate        # sous Windows : .venv\Scripts\activate
pip install mlflow scikit-learn pandas numpy
```

## Lancer les exemples

Suivez l'ordre des numéros pour progresser :

```bash
uv run python src/01_tracking_autolog.py
uv run python src/02_experimentations.py
uv run python src/03_models.py
uv run python src/04_registry_alias.py
uv run python src/05_genai_tracing.py
```

Si vous n'utilisez pas uv, retirez simplement le préfixe `uv run` (avec l'environnement activé).

Les résultats sont enregistrés dans une base locale `mlflow.db`. Vous n'avez rien d'autre à configurer.

## Voir vos résultats dans l'interface MLflow

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Ouvrez ensuite l'adresse indiquée dans votre navigateur. Vous y retrouverez vos expérimentations, vos modèles enregistrés, leurs alias, et (pour l'exemple 5) l'onglet des traces.

## Lancer le projet reproductible

Le fichier `MLproject` décrit un entraînement paramétrable :

```bash
uv run python train.py --n-estimators 200 --max-depth 15
# ou, via MLflow :
uv run mlflow run . -P n_estimators=200 -P max_depth=15
```

> **Deux fichiers d'environnement, deux rôles.** Le `pyproject.toml` (avec `uv.lock`) sert à installer et lancer les scripts de ce dépôt. Le fichier `python_env.yaml` est, lui, utilisé uniquement par le composant MLflow Projects pour recréer son propre environnement quand vous lancez `mlflow run`. Les deux coexistent normalement.

## Aller plus loin

Cet exemple couvre l'essentiel, mais MLflow va beaucoup plus loin. Si vous voulez monter en compétences sur le MLOps et l'industrialisation de vos modèles, Stat4decision propose des formations pratiques en petits groupes. Voir [nos formations data](https://www.stat4decision.com/fr/formations/) ou [nous contacter](https://www.stat4decision.com/fr/contactez-nous/).

## Licence

Code distribué sous licence MIT. Vous êtes libre de le réutiliser et de l'adapter.
