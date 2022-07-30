# IPS_anomaly_detection_XAI_flask_docker

[IPS (Intrusion Prevention System) Detection - Attack / Normal]

- Data: IPS Payload
- Feature create: PySpark (Spark SQL)
- Algorithm: CatBoost (SQL feature), LightGBM (TF-IDF Text feature)
- XAI: SHAP's force plot (SQL feature, TF-IDF Text feature), LIME's TextExplainer (TF-IDF Text feature)
- Deployment: Flask
- ssdeep (fuzzy hash similarity measure) based payload calculate & PostgreSQL connect


- TO DO 1: LightGBM model improving (special character & TF-IDF Text feature)
- TO DO 2: Docker application
