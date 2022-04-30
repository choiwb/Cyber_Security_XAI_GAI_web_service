# IPS_model_flask_server_deployment

[IPS (Intrusion Prevention System) Detection - Attack / Normal]

- Data: IPS Payload
- Feature create: PySpark (Spark SQL)
- Algorithm: LightGBM (SQL feature), CatBoost (Text feature)
- XAI: SHAP's force plot (SQL feature), LIME's TextExplainer (Text feature)
- Deployment: Flask
- ssdeep (fuzzy hash similarity measure) based payload calculate & PostgreSQL connect


- TO DO 1: CatBoost (Text feature) TF-IDF research
- TO DO 2: Docker application
