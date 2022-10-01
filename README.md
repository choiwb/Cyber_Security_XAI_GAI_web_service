# IPS_anomaly_detection_XAI_flask_docker

[IPS (Intrusion Prevention System) Detection - Attack / Normal]

- Data: IPS Payload
- Feature create: PySpark (Spark SQL)
- Algorithm: LightGBM ((SQL feature, TF-IDF Text feature)
- XAI: SHAP's force plot (SQL feature, TF-IDF Text feature), LIME's TextExplainer (TF-IDF Text feature), SHAP's text plot (BERT Algorithm feature)
- Deployment: Flask
- ssdeep (fuzzy hash similarity measure) based payload calculate & PostgreSQL connect

- TO DO 1: Domain Signature pattern highlighting and method matching
- TO DO 2: BERT based Transfer Learning using IPS paylaod
- TO DO 3: Docker application
