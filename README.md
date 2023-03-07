# IPS_anomaly_detection_XAI_flask_docker

[IPS (Intrusion Prevention System) Detection - anomalies / normal]

- Data: IPS Payload
- Feature create: PySpark (Spark SQL)
- Algorithm: LightGBM (SQL feature, TF-IDF Text feature)
- XAI: SHAP's force plot (SQL feature, TF-IDF Text feature), LIME's TextExplainer (TF-IDF Text feature), SHAP's text plot (BERT based Transfer Learning using IPS paylaod)
- Deployment: Flask
- ssdeep (fuzzy hash similarity measure) based payload calculate & PostgreSQL connect
- Domain Signature pattern and AI feature highlighting after pattern method matching
- Feature importance Summary about predict data
- WAF & WEB LOG (APACHE or NGINX or IIS) Parsing
- Docker application
- DistilBERT Transfer Learning (adding cyber security domain word)
- OpenAI ChatGPT API (gpt-3.5-turbo) based XAI analysis
-----
- TO DO 1: Mitre Att&ck Matrix mapping Tactic & T-ID based multi classification
- TO DO 2: DistilBERT (task: question-answering) based fine tuning like SQUAD dataset format
- TO DO 3: OpenAI GPT (https://huggingface.co/openai-gpt) based cyber security Chatbot R&D

