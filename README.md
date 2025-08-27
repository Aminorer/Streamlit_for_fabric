# Streamlit for Fabric

Cette application Streamlit interagit avec des bases de données SQL pour afficher des historiques et des prédictions de stocks.

## Variables d'environnement

Les informations de connexion sont fournies via des variables d'environnement (par exemple dans `secret.env`) :

- `SQL_USER` – Nom d'utilisateur SQL.
- `SQL_PASSWORD` – Mot de passe SQL.
- `SQL_SERVER` – Adresse du serveur SQL.
- `SQL_DRIVER` – Pilote ODBC à utiliser.
- `SQL_DATABASE_HIST` – Base contenant les données historiques.
- `SQL_DATABASE_PRED` – Base contenant les tables de prédiction.

Assurez-vous que ces variables sont définies avant de lancer l'application.
