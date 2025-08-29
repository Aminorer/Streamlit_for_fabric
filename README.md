# Streamlit for Fabric

Cette application Streamlit interagit avec des bases de données SQL pour afficher des historiques et des prédictions de stocks.

## Chargement des données

Les jeux de données factices ont été supprimés. Toutes les informations sont désormais récupérées directement depuis la base via les fonctions `load_hist_data` et `load_prediction_data` du module `db_utils`.
`load_hist_data` lit les volumes historiques tandis que `load_prediction_data` charge une table de prédictions en validant son nom et propose les mêmes filtres (marques, saisons, dimensions, dates).

```python
from db_utils import load_hist_data, load_prediction_data

df_hist = load_hist_data(brands=["MICHELIN"])
df_pred = load_prediction_data("pred_amz_man", brands=["MICHELIN"])
```

## Pages

L'application propose plusieurs pages accessibles depuis le menu :

- **Analyse comparative** (`pages/1_Analyse_Comparative.py`) — comparaison de l'historique avec plusieurs séries de prédictions et calcul de précision hebdomadaire.
- **Gestion des ruptures** (`pages/1_Gestion_Ruptures.py`) — suivi des dates de rupture principales et des recommandations de commande.
- **Analyse des prédictions** (`pages/1_Analyse_Prediction.py`) — répartition des statuts de stock et scores de criticité.
- **Analyse des prix** (`pages/2_Analyse_Prix.py`) — opportunités de marge et tendances de prix.
- **Volatilité & risques** (`pages/3_Volatilite_Risques.py`) — états de volatilité et alertes associées.

## Analyse comparative

Cette page confronte l'historique des volumes aux différentes séries de prédictions disponibles.
Elle permet de filtrer par marque, saison, dimension et période, puis :

- superpose les quantités historiques et chaque série `stock_prediction*` dans un graphique ;
- calcule la précision hebdomadaire (1 − MAPE) et affiche son évolution ;
- exporte les données combinées au format CSV.

Les légendes du graphique utilisent des icônes ("📈" pour l'historique,
"🔮" pour les prédictions) afin de faciliter la lecture. Elles peuvent être
désactivées en passant ``use_emoji=False`` à
``plot_historical_vs_multi_predictions`` ou via la case à cocher de l'interface.

Les tables de prédiction suivent un motif `pred_<plateforme>_<activité>_<YYYYMMDD>` :
la date correspond au mardi de génération de la semaine, les prévisions étant recalculées chaque mardi.
Les visualisations exploitent la palette de couleurs Wyz (`ASSOCIATED_COLORS`) définie dans `constants.py` pour assurer une identité visuelle homogène.

## Colonnes clés

Les tables de prédiction exposent notamment :

- `stock_status` : statut calculé du stock (OK, RUPTURE, etc.).
- `criticality_score` : niveau de criticité d'un article.
- `main_rupture_date`, `optimal_order_date`, `last_safe_order_date` : jalons de la chaîne d’approvisionnement.
- `order_recommendation`, `tension_days`, `recommended_volume` : informations de planification.
- `margin_opportunity_days` : nombre de jours avant une hausse de prix potentielle.
- `volatility_status`, `volatility_type`, `risk_level`, `anomaly_alert` : indicateurs de risque.

## Fabric / Delta

L'accès aux Lakehouses Fabric se fait via leurs SQL endpoints. Les variables
`SQL_SERVER_HIST` et `SQL_SERVER_PRED` doivent pointer respectivement vers les
endpoints des deux Lakehouses historiques et prédictifs. Utilisez un pilote
ODBC compatible, par exemple `ODBC Driver 18 for SQL Server`. Les tables Delta
sont exposées dans `INFORMATION_SCHEMA.TABLES` sous le schéma `dbo`.

## Configuration de développement

### Variables d'environnement requises

Les informations de connexion sont fournies via des variables d'environnement (par exemple dans `secret.env`).
Un fichier d'exemple `secret.env.example` est disponible à la racine : copiez-le en `secret.env` et renseignez vos propres valeurs, puis ajustez les variables suivantes :

- `SQL_USER` – Nom d'utilisateur SQL.
- `SQL_PASSWORD` – Mot de passe SQL.
- `SQL_DRIVER` – Pilote ODBC à utiliser.
- `SQL_SERVER_HIST` – Adresse du serveur hébergeant les données historiques.
- `SQL_DATABASE_HIST` – Base contenant les données historiques.
- `SQL_SERVER_PRED` – Adresse du serveur des tables de prédiction.
- `SQL_DATABASE_PRED` – Base contenant les tables de prédiction.

Les variables génériques `SQL_SERVER` et `SQL_DATABASE` ne sont plus prises en
charge ; vous devez définir explicitement les quatre variables ci-dessus.

### Conventions de nommage des tables

Les tables historiques suivent le motif `fullsize_stock_hist_%` et les tables de prédiction le motif `pred_%`. Les suffixes doivent correspondre pour former une paire cohérente, par exemple : `fullsize_stock_hist_amz_man` et `pred_amz_man`.

Les types d'activité valides sont :

- `man` – activités manufacturières ;
- `dis` – activités de distribution ;
- `mixte` – activités mixtes combinant les deux.

Ces valeurs forment la partie « activité » des noms de table.

### Création et alimentation des tables de test

Pour le développement ou les tests locaux, créez une paire de tables historique/prédiction respectant les conventions ci-dessus. Exemple minimal :

```sql
-- Table historique
CREATE TABLE dbo.fullsize_stock_hist_demo (
    date       DATE,
    brand      VARCHAR(50),
    sku        VARCHAR(50),
    stock_qty  INT
);
INSERT INTO dbo.fullsize_stock_hist_demo VALUES ('2024-01-01','MICHELIN','XYZ',100);

-- Table de prédiction correspondante
CREATE TABLE dbo.pred_demo (
    date              DATE,
    brand             VARCHAR(50),
    sku               VARCHAR(50),
    stock_qty         INT,
    stock_status      VARCHAR(20),
    criticality_score FLOAT
);
INSERT INTO dbo.pred_demo VALUES ('2024-01-01','MICHELIN','XYZ',100,'OK',0.1);
```

Ces tables de test peuvent ensuite être interrogées via les fonctions `load_hist_data` et `load_prediction_data`.

## Tests

Executez la suite de tests avec [pytest](https://pytest.org/):

```bash
pytest tests/
```
