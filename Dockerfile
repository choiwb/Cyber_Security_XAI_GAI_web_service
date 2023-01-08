# Use python as base image
FROM ubuntu:20.04
FROM python:3.8.12


# Use working directory 
# WORKDIR /Users/choiwb/Python_projects/이글루시큐리티_DSS_표준모델_Web_API/DSS_IPS_flask_server
WORKDIR /app

# openjdk-11 설치
RUN apt-get clean && apt-get -y update && apt-get -y upgrade
RUN apt-get -y install openjdk-11-jdk
# ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64/
# RUN export JAVA_HOME

COPY requirements.txt /app

COPY DSS_IPS_predict.py /app
COPY setting.py /app
COPY DSS_IPS_preprocess_sql_tfidf.py /app
COPY IPS_XAI_deploy_20221201_sql_tfidf.csv /app
COPY train_word_idf.csv /app

COPY templates /app/templates
COPY static /app/static
COPY saved_model /app/saved_model
COPY BERT_transfer_checkpoint /app/BERT_transfer_checkpoint


# Copy and install required packages
# RUN python -m pip install --upgrade pip
# Install requirements.txt
# RUN pip freeze > requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

# Install dependencies
RUN pip3 install --upgrade setuptools pip
RUN pip3 install -r requirements.txt

ENV FLASK_APP DSS_IPS_predict.py
ENV FLASK_RUN_HOST 0.0.0.0
ENV FLASK_RUN_PORT 17171


# flask app run
# CMD ["flask", "run", "--host", "--port"]

# Run flask app
CMD ["python3","DSS_IPS_predict.py"]


###################################
# docker build -t choiwb/dss_ips_ml_app .
# docker run -d -p 17171:17171 choiwb/dss_ips_ml_app