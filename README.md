# Cyber_Security_XAI_GAI_web_service

IPS & WAF Detection - attack / normal  
WEB Log Detection - SQL Injection, Command Injection, XSS, normal
-----
- Data: IPS & WAF Payload, WEB Log
- Feature create: PySpark (Spark SQL)
- Algorithm: LightGBM
- XAI: Shapley value based R&D
- Deployment: Flask & Docker
- Domain Signature pattern and AI feature highlighting after pattern method matching
- WAF & WEB LOG (APACHE or NGINX or IIS) Parsing
- WEB LOG based user-agent application, normal_bot_crawler, bad_bot_crawler classification (referenced https://user-agents.net/, https://github.com/mitchellkrogza/nginx-ultimate-bad-bot-blocker/blob/master/_generator_lists/bad-user-agents.list) & Start IP based country name connection (referenced https://dev.maxmind.com/geoip/geolite2-free-geolocation-data)
-----
- DistilBERT Transfer Learning (adding cyber security domain word)
- DistilBERT (task: question-answering) based fine tuning like SQUAD dataset format
- DistilBART based text-summarization
- OpenAI API (gpt-3.5-turbo & gpt-4) based XAI analysis
- Google API (PALM) test
- Cerebras GPT (https://huggingface.co/cerebras/Cerebras-GPT-111M) based Cyber Security domain fine tuning and Gradio based deployment
-----
- TO DO 1: Domain ML feature + Automatically generated Feature (ex, TF-IDF, ...) base LightGBM
- TO DO 2: Deep Learning XAI (shap.text_plot) customization
- TO DO 3: Polyglot-ko-1.3B (https://huggingface.co/EleutherAI/polyglot-ko-1.3b) based cyber security Chatbot R&D 

