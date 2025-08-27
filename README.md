# Streamlit for Fabric

Cette application Streamlit interagit avec des bases de données SQL pour afficher des historiques et des prédictions de stocks.

## Chargement des données

Les jeux de données factices ont été supprimés. Toutes les informations sont désormais récupérées directement depuis la base via les fonctions `load_hist_data` et `load_prediction_data` du module `db_utils`.
`load_hist_data` lit les volumes historiques tandis que `load_prediction_data` charge une table de prédictions whitelistée (`ALLOWED_TABLES`) et propose les mêmes filtres (marques, saisons, dimensions, dates).

```python
from db_utils import load_hist_data, load_prediction_data

df_hist = load_hist_data(brands=["MICHELIN"])
df_pred = load_prediction_data("pred_amz_man", brands=["MICHELIN"])
```

## Pages

L'application propose plusieurs pages accessibles depuis le menu :

- **Gestion des ruptures** (`pages/1_Gestion_Ruptures.py`) — suivi des dates de rupture principales et des recommandations de commande.
- **Analyse des prédictions** (`pages/1_Analyse_Prediction.py`) — répartition des statuts de stock et scores de criticité.
- **Analyse des prix** (`pages/2_Analyse_Prix.py`) — opportunités de marge et tendances de prix.
- **Volatilité & risques** (`pages/3_Volatilite_Risques.py`) — états de volatilité et alertes associées.

## Colonnes clés

Les tables de prédiction exposent notamment :

- `stock_status` : statut calculé du stock (OK, RUPTURE, etc.).
- `criticality_score` : niveau de criticité d'un article.
- `main_rupture_date`, `optimal_order_date`, `last_safe_order_date` : jalons de la chaîne d’approvisionnement.
- `order_recommendation`, `tension_days`, `recommended_volume` : informations de planification.
- `margin_opportunity_days` : nombre de jours avant une hausse de prix potentielle.
- `volatility_status`, `volatility_type`, `risk_level`, `anomaly_alert` : indicateurs de risque.

## Variables d'environnement

Les informations de connexion sont fournies via des variables d'environnement (par exemple dans `secret.env`) :

 - `SQL_USER` – Nom d'utilisateur SQL.
 - `SQL_PASSWORD` – Mot de passe SQL.
 - `SQL_DRIVER` – Pilote ODBC à utiliser.
 - `SQL_SERVER_HIST` – Adresse du serveur hébergeant les données historiques.
 - `SQL_DATABASE_HIST` – Base contenant les données historiques.
 - `SQL_SERVER_PRED` – Adresse du serveur des tables de prédiction.
 - `SQL_DATABASE_PRED` – Base contenant les tables de prédiction.
 - `ALLOWED_TABLES` – Liste (séparée par des virgules) des tables autorisées.
  Elle doit inclure chaque table SQL qu'on souhaite interroger, sinon les fonctions de chargement refuseront la requête.

Assurez-vous que ces variables sont définies avant de lancer l'application et que `ALLOWED_TABLES` contient bien la whiteliste des tables disponibles.
