# IPS_anomaly_detection_XAI_flask_docker

[IPS (Intrusion Prevention System) Detection - Attack / Normal]

- Data: IPS Payload
- Feature create: PySpark (Spark SQL)
- Algorithm: CatBoost (SQL feature), CatBoost (Text feature)
- XAI: SHAP's force plot (SQL feature), LIME's TextExplainer (Text feature)
- Deployment: Flask
- ssdeep (fuzzy hash similarity measure) based payload calculate & PostgreSQL connect


- TO DO 1: CatBoost (Text feature) TF-IDF research
- TO DO 2: Docker application
