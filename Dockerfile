
# Use python as base image
FROM ubuntu:22.04.3
FROM python:3.11.5


# Use working directory 
WORKDIR /app

# openjdk-11 설치
RUN apt-get clean && apt-get -y update && apt-get -y upgrade
RUN apt-get -y install openjdk-17-jdk

COPY requirements.txt /app

# IPython 설치
RUN pip3 install ipython

COPY runserver.py /app
COPY setting.py /app
COPY shap_explainer_save.py /app

COPY templates /app/templates
COPY static /app/static
COPY save_model /app/save_model
COPY save_model/IPS_KISA_DL_202311 /app/save_model/IPS_KISA_DL_202311
COPY save_model/WAF_KISA_DL_202311 /app/save_model/WAF_KISA_DL_202311
COPY save_model/WEB_KISA_DL_202311 /app/save_model/WEB_KISA_DL_202311
COPY chat_gpt_context /app/chat_gpt_context

# Install dependencies
RUN pip3 install --upgrade setuptools pip
RUN pip3 install -r requirements.txt

ENV FLASK_APP runserver.py
# ENV FLASK_ENV development
ENV FLASK_RUN_HOST 0.0.0.0
ENV FLASK_RUN_PORT 17171

# Run flask app
CMD ["python3","runserver.py"]


###################################
# docker build -t choiwb/ml_xai_gai_web .
# docker run -d -p 17171:17171 choiwb/ml_xai_gai_web
