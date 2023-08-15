


import re
import pickle
import itertools
import pandas as pd
import numpy as np
import scipy as sp
from sklearn.feature_extraction.text import CountVectorizer
import torch
import shap
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification


SERVER_IP = '0.0.0.0'
PORT = 17171


# 2023/08/17 IPS 모델 - Light GBM
IPS_model_path = 'save_model/IPS_ML_20230817.pkl'
# 위 모델을, SHAP의 TreeExplainer 연산 및 저장
IPS_explainer_path = 'save_model/IPS_ML_XAI_20230817.pkl'

IPS_model = pickle.load(open(IPS_model_path, 'rb'))


# 2023/08/17 WAF 모델 - Light GBM
WAF_model_path = 'save_model/WAF_ML_20230817.pkl'
# 2023/06/22 WAF 모델 - Light GBM (Spark SQL 피처 & TF-IDF 피처)
# new_WAF_model_path = 'save_model/DSS_WAF_sql_tfidf_LGB_20230622.pkl'
# 위 모델의 TF-IDF 단어 호출
# WAF_tfidf_word_path = 'save_model/waf_tfidf_word.csv'


# 위 모델을, SHAP의 TreeExplainer 연산 및 저장
WAF_explainer_path = 'save_model/WAF_ML_XAI_20230817.pkl'
# WAF_explainer_path = 'save_model/DSS_WAF_sql_tfidf_LGB_explainer_20230622.pkl'

WAF_model = pickle.load(open(WAF_model_path, 'rb'))


####################################################################################
# IPS 딥러닝 모델 호출
IPS_DL_path = 'save_model/IPS_DL_20230817'

IPS_DL_model = AutoModelForSequenceClassification.from_pretrained(IPS_DL_path)
IPS_DL_tokenizer = AutoTokenizer.from_pretrained(IPS_DL_path)
IPS_DL_model.eval()

# XAI 서버에 NVIDIA 드라이버 설치 시, 각 모델 별, (IPS, WAF, WEB) 에 대한 디바이스 지정 필요해 보임 !!!!!!
# 드라이버 분할 지정하지 않는다면, 'cuda:0' 이나 'cuda' 해도 상관 없음.
ips_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# token 수 512개 제한 - IPS / WAF/ WEB 모두 DistilBERT의 기본 토큰 수로 학습 되어있으므로 512개 제한
# 추 후, 장비 별, 모델 학습 시, 1024 개 이상으로 학습 필요해 보임. (학습 시, GPU 고려 또한 해야 함.)
max_length = 512
def ips_truncate_text(text, max_length=max_length):
    # CLS, SEP 토큰을 고려해야 하기 때문에 -2
    encoding = IPS_DL_tokenizer.encode_plus(text, max_length=max_length-2, truncation=True, padding='max_length')
    return IPS_DL_tokenizer.decode(encoding["input_ids"])

ips_dl_pipe = pipeline('text-classification', model=IPS_DL_model, tokenizer=IPS_DL_tokenizer, device=ips_device)
print('IPS 딥러닝 모델 디바이스: ', ips_dl_pipe.device)

# define a prediction function
def ips_bert_predict(x):
    tv = torch.tensor([IPS_DL_tokenizer.encode(v, padding='max_length', max_length=64, truncation=True) for v in x]).to(ips_device)

    # outputs = model(tv)[0].detach().cpu().numpy()
    outputs = IPS_DL_model(tv)[0].detach().cpu().numpy()

    scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
    val = sp.special.logit(scores[:,1]) # use one vs rest logit units

    return val

# payload의 특정 패턴 기준으로 분할 regex
ips_masker = shap.maskers.Text(tokenizer = r"(\s|%20|\+|\/|%2f|HTTP/1.1|\?|\n|\r|\t)")
# masker = shap.maskers.Text(tokenizer = r"(\s|%20|\+|%2f|HTTP/1.1|\?|\n|\r|\t)")

IPS_DL_XAI = shap.Explainer(ips_bert_predict, ips_masker)

####################################################################################

####################################################################################
# WAF 딥러닝 모델 호출
WAF_DL_path = 'save_model/WAF_DL_20230817'

WAF_DL_model = AutoModelForSequenceClassification.from_pretrained(WAF_DL_path)
WAF_DL_tokenizer = AutoTokenizer.from_pretrained(WAF_DL_path)
WAF_DL_model.eval()

# XAI 서버에 NVIDIA 드라이버 설치 시, 각 모델 별, (IPS, WAF, WEB) 에 대한 디바이스 지정 필요해 보임 !!!!!!
# 드라이버 분할 지정하지 않는다면, 'cuda:0' 이나 'cuda' 해도 상관 없음.
waf_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# token 수 512개 제한 - IPS / WAF/ WEB 모두 DistilBERT의 기본 토큰 수로 학습 되어있으므로 512개 제한
# 추 후, 장비 별, 모델 학습 시, 1024 개 이상으로 학습 필요해 보임. (학습 시, GPU 고려 또한 해야 함.)
max_length = 512
def waf_truncate_text(text, max_length=max_length):
    # CLS, SEP 토큰을 고려해야 하기 때문에 -2
    encoding = WAF_DL_tokenizer.encode_plus(text, max_length=max_length-2, truncation=True, padding='max_length')
    return WAF_DL_tokenizer.decode(encoding["input_ids"])

waf_dl_pipe = pipeline('text-classification', model=WAF_DL_model, tokenizer=WAF_DL_tokenizer, device=waf_device)
print('WAF 딥러닝 모델 디바이스: ', waf_dl_pipe.device)

# define a prediction function
def waf_bert_predict(x):
    tv = torch.tensor([WAF_DL_tokenizer.encode(v, padding='max_length', max_length=64, truncation=True) for v in x]).to(waf_device)

    # outputs = model(tv)[0].detach().cpu().numpy()
    outputs = WAF_DL_model(tv)[0].detach().cpu().numpy()

    scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
    val = sp.special.logit(scores[:,1]) # use one vs rest logit units

    return val

# payload의 특정 패턴 기준으로 분할 regex
waf_masker = shap.maskers.Text(tokenizer = r"(\s|%20|\+|\/|%2f|HTTP/1.1|\?|\n|\r|\t)")
# masker = shap.maskers.Text(tokenizer = r"(\s|%20|\+|%2f|HTTP/1.1|\?|\n|\r|\t)")

WAF_DL_XAI = shap.Explainer(waf_bert_predict, waf_masker)
####################################################################################

####################################################################################
# WEB 딥러닝 모델 호출
WEB_DL_path = 'save_model/WEB_DL_20230817'

WEB_DL_model = AutoModelForSequenceClassification.from_pretrained(WEB_DL_path)
WEB_DL_tokenizer = AutoTokenizer.from_pretrained(WEB_DL_path)
WEB_DL_model.eval()

# XAI 서버에 NVIDIA 드라이버 설치 시, 각 모델 별, (IPS, WAF, WEB) 에 대한 디바이스 지정 필요해 보임 !!!!!!
# 드라이버 분할 지정하지 않는다면, 'cuda:0' 이나 'cuda' 해도 상관 없음.
web_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# token 수 512개 제한 - IPS / WAF/ WEB 모두 DistilBERT의 기본 토큰 수로 학습 되어있으므로 512개 제한
# 추 후, 장비 별, 모델 학습 시, 1024 개 이상으로 학습 필요해 보임. (학습 시, GPU 고려 또한 해야 함.)
max_length = 512
def web_truncate_text(text, max_length=max_length):
    # CLS, SEP 토큰을 고려해야 하기 때문에 -2
    encoding = WEB_DL_tokenizer.encode_plus(text, max_length=max_length-2, truncation=True, padding='max_length')
    return WEB_DL_tokenizer.decode(encoding["input_ids"])

web_dl_pipe = pipeline('text-classification', model=WEB_DL_model, tokenizer=WEB_DL_tokenizer, device=web_device)
print('WEB 딥러닝 모델 디바이스: ', web_dl_pipe.device)

# define a prediction function
def web_bert_predict(x, pipe_result_label):
    tv = torch.tensor([WEB_DL_tokenizer.encode(v, padding='max_length', max_length=64, truncation=True) for v in x]).to(web_device)

    # outputs = model(tv)[0].detach().cpu().numpy()
    outputs = WEB_DL_model(tv)[0].detach().cpu().numpy()
    scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T

    # 라벨 별 인덱스 매칭
    if pipe_result_label == 'CMD Injection':
        pred_label_index = 0
    elif pipe_result_label == 'SQL Injection':
        pred_label_index = 2
    elif pipe_result_label == 'XSS':
        pred_label_index = 3
    else:
        # '정상'
        pred_label_index = 1
    
    val = sp.special.logit(scores[:,pred_label_index]) # use one vs rest logit units

    return val

# payload의 특정 패턴 기준으로 분할 regex
web_masker = shap.maskers.Text(tokenizer = r"(\s|%20|\+|\/|%2f|HTTP/1.1|\?|\n|\r|\t)")
# masker = shap.maskers.Text(tokenizer = r"(\s|%20|\+|%2f|HTTP/1.1|\?|\n|\r|\t)")

# 공격/정상인 경우, bert_predict 함수에 '1'로 index를 지정할 수 있으나, web log의 경우, 다중 분류이므로, 예측 값에 따라 달라지므로,
# runserver.py 에서 pipe_result_label 에 따라서, 해당 라벨에 대한 index를 지정해 줌.
# WEB_DL_XAI = shap.Explainer(lambda x: web_bert_predict(x, pipe_result_label), masker)

####################################################################################


# 2023/08/17 WEB 모델 - Light GBM
WEB_model_path = 'save_model/WEB_ML_20230817.pkl'
# 위 모델을, SHAP의 TreeExplainer 연산 및 저장
WEB_explainer_path = 'save_model/WEB_ML_XAI_20230817.pkl'

WEB_model = pickle.load(open(WEB_model_path, 'rb'))


# 2023/04/13 WEB User-Agent 모델 - Light GBM
WEB_useragent_model_path = 'save_model/DSS_WEB_useragent_LGB_20230413.pkl'
# 위 모델의 TF-IDF 단어 호출
WEB_useragent_tfidf_word_path = 'save_model/useragent_tfidf_word.csv'

WEB_useragent_model = pickle.load(open(WEB_useragent_model_path, 'rb'))
WEB_useragent_tfidf_word = pd.read_csv(WEB_useragent_tfidf_word_path)
WEB_useragent_tfidf_word = WEB_useragent_tfidf_word.sort_values(by = 'word', ascending = True)

tfidf_feature = WEB_useragent_tfidf_word['feature'].tolist()
tfidf_word = WEB_useragent_tfidf_word['word'].tolist()
tfidf_value = WEB_useragent_tfidf_word['IDF'].tolist()

sep_list = [' ']
sep_str = '|'.join(sep_list)

vectorizer = CountVectorizer(lowercase = True,
                             tokenizer = lambda x: re.split(sep_str, x),
                              vocabulary = tfidf_word)


WAF_tfidf_word = pd.read_csv(WAF_tfidf_word_path)
WAF_tfidf_word = WAF_tfidf_word.sort_values(by = 'word', ascending = True)

tfidf_feature_waf = WAF_tfidf_word['feature'].tolist()
tfidf_word_waf = WAF_tfidf_word['word'].tolist()
tfidf_value_waf = WAF_tfidf_word['IDF'].tolist()

sep_list_waf = [' ', '%20', '\\+', '\\/', '%2f', 'HTTP/1.1', '\\?', '\\n', '\\r', '\\t']
sep_str_waf = '|'.join(sep_list_waf)

vectorizer_waf = CountVectorizer(lowercase = True,
                             tokenizer = lambda x: re.split(sep_str_waf, x),
                            vocabulary = tfidf_word_waf)

# GeoIP2의 국가명 조회 DB 경로 - 20230419 업데이트 기준
geoip_country_db_path = 'save_model/GeoLite2_Country_20230419.mmdb'


ips_query = """
    
    SELECT
   
        IF((SIZE(SPLIT(REGEXP_REPLACE(payload, '\\n|\\r|\\t', ' '), 'GET(.*?)HTTP/1.')) -1)
            + (SIZE(SPLIT(REGEXP_REPLACE(payload, '\\n|\\r|\\t', ' '), 'POST(.*?)HTTP/1.')) -1)
            + (SIZE(SPLIT(REGEXP_REPLACE(payload, '\\n|\\r|\\t', ' '), 'HEAD(.*?)HTTP/1.')) -1)
            + (SIZE(SPLIT(REGEXP_REPLACE(payload, '\\n|\\r|\\t', ' '), 'OPTION(.*?)HTTP/1.')) -1) >= 2
            , 0, 1) AS ips_payload_whitelist,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'currentsetting(.*?)htm') )>0
            OR INT(RLIKE(LOWER(payload), 'get [\\/]hnap1') )>0
            OR INT(RLIKE(LOWER(payload), 'administrator') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'admin(.*?)serv(.*?)admpw') )>0
            , 1, 0) AS ips_payload_auth_comb,

        IF(INT(RLIKE(LOWER(payload), 'aaaaaaaaaa') )>0
            OR INT(RLIKE(LOWER(payload), 'cacacacaca') )>0
            , 1, 0) AS ips_payload_bof_comb,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  'wget[\\+](.*?)ttp') )>0
			OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'wget[\\-]c(.*?)ttp') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'wget[\\%]20(.*?)ttp') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'wget (.*?)ttp') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'wget[\\$][\\{](.*?)ttp') )>0
			OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  'wget[\\%](.*?)ttp') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  'wget[\\$](.*?)ttp') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'rm(.*?)[\\-]rf') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'cd (.*?)tmp') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'cd[\\%]20(.*?)tmp') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'cd[\\+](.*?)tmp') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'cd[\\$][\\{](.*?)tmp') )>0
			OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  'cd(.*?)[\\/]tmp') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  'chmod[\\+](.*?)777') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  'chmod[\\%](.*?)777') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'chmod[\\$](.*?)777') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'chmod[\\(][\\$](.*?)777') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'chmod[\\+](.*?)777') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'chmod (.*?)777') )>0
            , 1, 0) AS ips_payload_cmd_01_comb,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'cmd(.*?)open') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'echo(.*?)shellshock') )>0
            OR INT(RLIKE(LOWER(payload), 'powershell'))>0
            OR INT(RLIKE(LOWER(payload), '[\\/]tcsh'))>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'api(.*?)ping') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'get(.*?)ping') )>0
            , 1, 0) AS ips_payload_cmd_02_comb,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'eval[\\(]') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'eval[\\/][\\*]') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'eval[\\%]') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'eval[\\|]') )>0
			OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'eval[\\-]') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'php[\\/]eval') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'getruntime(.*?)exec') )>0
            , 1, 0) AS ips_payload_code_comb,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'current[\\_]config(.*?)passwd') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'well(.*?)known') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'backup(.*?)sql') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'robots(.*?)txt') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'etc(.*?)passwd') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'etc(.*?)shadow') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'bash(.*?)history') )>0
            , 1, 0) AS ips_payload_dir_01_comb,

        IF(INT(RLIKE(LOWER(payload), 'htaccess') )>0
            OR INT(RLIKE(LOWER(payload), 'htpasswd') )>0
            OR INT(RLIKE(LOWER(payload), '[\\.]env'))>0
            OR INT(RLIKE(LOWER(payload), 'access') )>0
            OR INT(RLIKE(LOWER(payload), '[\\/]bash') )>0
            , 1, 0) AS ips_payload_dir_02_comb,

        (SIZE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), '[\\.][\\.][\\/]')) -1)
            + (SIZE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), '[\\.][\\.][\\%]2f')) -1)
            + (SIZE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), '[\\%]2e[\\%]2e[\\%]2f')) -1)
            AS ips_payload_dir_count,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'cgi(.*?)bin') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'cgi(.*?)cgi') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'count(.*?)cgi(.*?)http') )>0
            OR INT(RLIKE(LOWER(payload), '[\\.]cgi') )>0
            OR INT(RLIKE(LOWER(payload), 'search(.*?)cgi') )>0
            OR INT(RLIKE(LOWER(payload), 'bbs(.*?)forum(.*?)cgi') )>0
            OR INT(RLIKE(LOWER(payload), 'web(.*?)store(.*?)cgi') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'count(.*?)cgi(.*?)http') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'msadc(.*?)dll(.*?)http') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'cgi(.*?)cgi(.*?)cgimail(.*?)exe') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'cgi(.*?)cgi(.*?)fpcount(.*?)exe') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'cgi(.*?)cgi(.*?)rguest(.*?)exe') )>0
            , 1, 0) AS ips_payload_cgi_comb,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'wp[\\-]login') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'wp[\\-]content') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'wp[\\-]include') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'wp[\\-]config') )>0
            , 1, 0) AS ips_payload_wp_comb,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'this(.*?)program(.*?)can') )>0
            OR INT(RLIKE(LOWER(payload), '80040e07(.*?)font') )>0
            OR INT(RLIKE(LOWER(payload), '80040e14(.*?)font') )>0
            , 1, 0) AS ips_payload_error_comb,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'filename(.*?)asp') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'filename(.*?)jsp') )>0
            OR INT(RLIKE(LOWER(payload), '[\\/]a[\\.]jsp') )>0
            OR INT(RLIKE(LOWER(payload), '[\\.]asp[\\;][\\.]jpg') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'upload(.*?)asp') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'fckeditor(.*?)filemanager') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'manager(.*?)html') )>0
            OR INT(RLIKE(LOWER(payload), '[\\.]mdb') )>0
            , 1, 0) AS ips_payload_file_comb,

        IF(INT(RLIKE(LOWER(payload), 'delete [\\/]') )>0
            OR INT(RLIKE(LOWER(payload), 'put [\\/]') )>0
            , 1, 0) AS ips_payload_http_method_comb,

        IF(INT(RLIKE(LOWER(payload), 'mozi[\\.]') )>0
            , 1, 0) AS ips_payload_malware_comb,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'apache(.*?)struts') )>0
            OR INSTR(LOWER(payload), 'jdatabasedrivermysqli')>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'jndi(.*?)dap') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),'jndi(.*?)dns') )>0
            , 1, 0) AS ips_payload_rce_comb,

        IF( INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  'select(.*?)from') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  'select(.*?)count') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  'select(.*?)distinct') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  'union(.*?)select') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  'select(.*?)extractvalue(.*?)xmltype') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  'and(.*?)select') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  'from(.*?)generate(.*?)series') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  'from(.*?)group(.*?)by') )>0
            , 1, 0) AS ips_payload_sql_01_comb,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  'case(.*?)when') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  'then(.*?)else') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  'like[\\%]') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  'sleep[\\%]') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  'sleep[\\(]') )>0
			OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'sleep[\\+]') )>0
			OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'drop[\\%]') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  'drop[\\+]table') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  'waitfor(.*?)delay') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  'db(.*?)sql(.*?)server') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  'cast(.*?)chr') )>0
			OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'cast(.*?)chr[\\%]') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'cast(.*?)chr[\\(]') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'cast(.*?)char[\\%]') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'cast(.*?)char[\\(]') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  'upper(.*?)xmltype') )>0
                , 1, 0) AS ips_payload_sql_02_comb,

        IF((INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'bingbot') )>0)
            OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'user(.*?)agent(.*?)zgrab') )>0)
            OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'user(.*?)agent(.*?)nmap') )>0)
            OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'user(.*?)agent(.*?)dirbuster') )>0)
            OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'user(.*?)agent(.*?)ahrefsbot') )>0)
            OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'user(.*?)agent(.*?)baiduspider') )>0)
            OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'user(.*?)agent(.*?)mj12bot') )>0)
            OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'user(.*?)agent(.*?)petalbot') )>0)
            OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'user(.*?)agent(.*?)curl[\\/]') )>0)
            OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'user(.*?)agent(.*?)semrushbot') )>0)
            OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'user(.*?)agent(.*?)masscan') )>0)
            OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'user(.*?)agent(.*?)sqlmap') )>0)
            OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'user(.*?)agent(.*?)urlgrabber(.*?)yum') )>0)
            OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'user(.*?)agent(.*?)zmeu') )>0)
            , 1, 0) AS ips_payload_useragent_comb,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'php(.*?)echo') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'admin(.*?)php') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'script(.*?)setup(.*?)php') )>0
            OR INT(RLIKE(LOWER(payload), 'phpinfo') )>0
            OR INT(RLIKE(LOWER(payload), 'phpmyadmin') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'php(.*?)create(.*?)function') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'php(.?)content(.*?)type') )>0
            OR INT(RLIKE(LOWER(payload), 'md5[\\(]') )>0
            OR INT(RLIKE(LOWER(payload), 'md5[\\%]') )>0
            OR INT(RLIKE(LOWER(payload),'md5[\\"]') )>0
            OR INT(RLIKE(LOWER(payload), '[\\"]md5') )>0
            OR INT(RLIKE(LOWER(payload), '[\\']md5') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'upload(.*?)php') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'sqlexec(.*?)php') )>0
            , 1, 0) AS ips_payload_php_comb,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'script(.*?)alert') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'onerror(.*?)alert') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'document(.*?)createelement') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'document(.*?)forms') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'document(.*?)location') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'document(.*?)open') )>0
            , 1, 0) AS ips_payload_xss_comb,

        SIZE(SPLIT(IF(ISNULL(payload), '', payload), '[\\&]'))-1 AS ips_payload_ampersand_count,

        SIZE(SPLIT(IF(ISNULL(payload), '', payload), '[\\;]'))-1 AS ips_payload_semicolon_count


    FROM table
    """

waf_query = """
    
    SELECT
   
        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'currentsetting(.*?)htm') )>0
            OR INT(RLIKE(LOWER(payload), 'get [\\/]hnap1') )>0
            OR INT(RLIKE(LOWER(payload), 'administrator') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'admin(.*?)serv(.*?)admpw') )>0
            , 1, 0) AS waf_payload_auth_comb,

        IF(INT(RLIKE(LOWER(payload), 'aaaaaaaaaa') )>0
            OR INT(RLIKE(LOWER(payload), 'cacacacaca') )>0
            , 1, 0) AS waf_payload_bof_comb,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  'wget[\\+](.*?)ttp') )>0
			OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'wget[\\-]c(.*?)ttp') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'wget[\\%]20(.*?)ttp') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'wget (.*?)ttp') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'wget[\\$][\\{](.*?)ttp') )>0
			OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  'wget[\\%](.*?)ttp') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  'wget[\\$](.*?)ttp') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'rm(.*?)[\\-]rf') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'cd (.*?)tmp') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'cd[\\%]20(.*?)tmp') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'cd[\\+](.*?)tmp') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'cd[\\$][\\{](.*?)tmp') )>0
			OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  'cd(.*?)[\\/]tmp') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  'chmod[\\+](.*?)777') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  'chmod[\\%](.*?)777') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'chmod[\\$](.*?)777') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'chmod[\\(][\\$](.*?)777') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'chmod[\\+](.*?)777') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'chmod (.*?)777') )>0
            , 1, 0) AS waf_payload_cmd_01_comb,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'cmd(.*?)open') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'echo(.*?)shellshock') )>0
            OR INT(RLIKE(LOWER(payload), 'powershell'))>0
            OR INT(RLIKE(LOWER(payload), '[\\/]tcsh'))>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'api(.*?)ping') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'get(.*?)ping') )>0
            , 1, 0) AS waf_payload_cmd_02_comb,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'eval[\\(]') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'eval[\\/][\\*]') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'eval[\\%]') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'eval[\\|]') )>0
			OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'eval[\\-]') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'php[\\/]eval') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'getruntime(.*?)exec') )>0
            , 1, 0) AS waf_payload_code_comb,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'current[\\_]config(.*?)passwd') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'well(.*?)known') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'backup(.*?)sql') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'robots(.*?)txt') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'etc(.*?)passwd') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'etc(.*?)shadow') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'bash(.*?)history') )>0
            , 1, 0) AS waf_payload_dir_01_comb,

        IF(INT(RLIKE(LOWER(payload), 'htaccess') )>0
            OR INT(RLIKE(LOWER(payload), 'htpasswd') )>0
            OR INT(RLIKE(LOWER(payload), '[\\.]env'))>0
            OR INT(RLIKE(LOWER(payload), 'access') )>0
            OR INT(RLIKE(LOWER(payload), '[\\/]bash') )>0
            , 1, 0) AS waf_payload_dir_02_comb,

        (SIZE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), '[\\.][\\.][\\/]')) -1)
            + (SIZE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), '[\\.][\\.][\\%]2f')) -1)
            + (SIZE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), '[\\%]2e[\\%]2e[\\%]2f')) -1)
            AS waf_payload_dir_count,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'cgi(.*?)bin') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'cgi(.*?)cgi') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'count(.*?)cgi(.*?)http') )>0
            OR INT(RLIKE(LOWER(payload), '[\\.]cgi') )>0
            OR INT(RLIKE(LOWER(payload), 'search(.*?)cgi') )>0
            OR INT(RLIKE(LOWER(payload), 'bbs(.*?)forum(.*?)cgi') )>0
            OR INT(RLIKE(LOWER(payload), 'web(.*?)store(.*?)cgi') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'count(.*?)cgi(.*?)http') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'msadc(.*?)dll(.*?)http') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'cgi(.*?)cgi(.*?)cgimail(.*?)exe') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'cgi(.*?)cgi(.*?)fpcount(.*?)exe') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'cgi(.*?)cgi(.*?)rguest(.*?)exe') )>0
            , 1, 0) AS waf_payload_cgi_comb,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'wp[\\-]login') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'wp[\\-]content') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'wp[\\-]include') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'wp[\\-]config') )>0
            , 1, 0) AS waf_payload_wp_comb,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'this(.*?)program(.*?)can') )>0
            OR INT(RLIKE(LOWER(payload), '80040e07(.*?)font') )>0
            OR INT(RLIKE(LOWER(payload), '80040e14(.*?)font') )>0
            , 1, 0) AS waf_payload_error_comb,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'filename(.*?)asp') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'filename(.*?)jsp') )>0
            OR INT(RLIKE(LOWER(payload), '[\\/]a[\\.]jsp') )>0
            OR INT(RLIKE(LOWER(payload), '[\\.]asp[\\;][\\.]jpg') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'upload(.*?)asp') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'fckeditor(.*?)filemanager') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'manager(.*?)html') )>0
            OR INT(RLIKE(LOWER(payload), '[\\.]mdb') )>0
            , 1, 0) AS waf_payload_file_comb,

        IF(INT(RLIKE(LOWER(payload), 'delete [\\/]') )>0
            OR INT(RLIKE(LOWER(payload), 'put [\\/]') )>0
            , 1, 0) AS waf_payload_http_method_comb,

        IF(INT(RLIKE(LOWER(payload), 'mozi[\\.]') )>0
            , 1, 0) AS waf_payload_malware_comb,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'apache(.*?)struts') )>0
            OR INSTR(LOWER(payload), 'jdatabasedrivermysqli')>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'jndi(.*?)dap') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),'jndi(.*?)dns') )>0
            , 1, 0) AS waf_payload_rce_comb,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'select(.*?)from') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'select(.*?)count') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'select(.*?)distinct') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'union(.*?)select') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'select(.*?)extractvalue(.*?)xmltype') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'and(.*?)select') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'from(.*?)generate(.*?)series') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'from(.*?)group(.*?)by') )>0
            , 1, 0) AS waf_payload_sql_01_comb,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  'case(.*?)when') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  'then(.*?)else') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  'like[\\%]') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  'sleep[\\%]') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  'sleep[\\(]') )>0
			OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'sleep[\\+]') )>0
			OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'drop[\\%]') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  'drop[\\+]table') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  'waitfor(.*?)delay') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  'db(.*?)sql(.*?)server') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  'cast(.*?)chr') )>0
			OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'cast(.*?)chr[\\%]') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'cast(.*?)chr[\\(]') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'cast(.*?)char[\\%]') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'cast(.*?)char[\\(]') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  'upper(.*?)xmltype') )>0
            , 1, 0) AS waf_payload_sql_02_comb,

        IF((INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'bingbot') )>0)
            OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'user(.*?)agent(.*?)zgrab') )>0)
            OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'user(.*?)agent(.*?)nmap') )>0)
            OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'user(.*?)agent(.*?)dirbuster') )>0)
            OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'user(.*?)agent(.*?)ahrefsbot') )>0)
            OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'user(.*?)agent(.*?)baiduspider') )>0)
            OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'user(.*?)agent(.*?)mj12bot') )>0)
            OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'user(.*?)agent(.*?)petalbot') )>0)
            OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'user(.*?)agent(.*?)curl[\\/]') )>0)
            OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'user(.*?)agent(.*?)semrushbot') )>0)
            OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'user(.*?)agent(.*?)masscan') )>0)
            OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'user(.*?)agent(.*?)sqlmap') )>0)
            OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'user(.*?)agent(.*?)urlgrabber(.*?)yum') )>0)
            OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'user(.*?)agent(.*?)zmeu') )>0)
            , 1, 0) AS waf_payload_useragent_comb,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'php(.*?)echo') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'admin(.*?)php') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'script(.*?)setup(.*?)php') )>0
            OR INT(RLIKE(LOWER(payload), 'phpinfo') )>0
            OR INT(RLIKE(LOWER(payload), 'phpmyadmin') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'php(.*?)create(.*?)function') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'php(.?)content(.*?)type') )>0
            OR INT(RLIKE(LOWER(payload), 'md5[\\(]') )>0
            OR INT(RLIKE(LOWER(payload), 'md5[\\%]') )>0
            OR INT(RLIKE(LOWER(payload),'md5[\\"]') )>0
            OR INT(RLIKE(LOWER(payload), '[\\"]md5') )>0
            OR INT(RLIKE(LOWER(payload), '[\\']md5') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'upload(.*?)php') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'sqlexec(.*?)php') )>0
            , 1, 0) AS waf_payload_php_comb,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'script(.*?)alert') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'onerror(.*?)alert') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'document(.*?)createelement') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'document(.*?)forms') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'document(.*?)location') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'document(.*?)open') )>0
            , 1, 0) AS waf_payload_xss_comb,

        SIZE(SPLIT(IF(ISNULL(payload), '', payload), '[\\&]'))-1 AS waf_payload_ampersand_count,

        SIZE(SPLIT(IF(ISNULL(payload), '', payload), '[\\;]'))-1 AS waf_payload_semicolon_count
        
    FROM table
    """




web_query = """

SELECT 

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '),  'select(.*?)from') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'select(.*?)count') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '),  'select(.*?)concat') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '),  'select(.*?)distinct') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'union(.*?)select') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '),  'select(.*?)extractvalue(.*?)xmltype') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '),  'from(.*?)generate(.*?)series') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '),  'from(.*?)group(.*?)by') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '),  'and(.*?)select') )>0
        ,1, 0) AS weblog_sql_01_comb,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '),  'case(.*?)when') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'then(.*?)else') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '),  'like[\\%]') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '),  'sleep[\\%]') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '),  'sleep[\\(]') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'sleep[\\+]') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '),  'delete(.*?)from') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'drop[\\%]') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '),  'drop[\\+]table') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '),  'cast(.*?)chr') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'cast(.*?)chr[\\%]') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'cast(.*?)chr[\\(]') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'cast(.*?)char[\\%]') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'cast(.*?)char[\\(]') )>0
        ,1, 0) AS weblog_sql_02_comb,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'where[\\_]framework') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'sql[\\_]server') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '),  'order[\\=]1') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '),  'id[\\=]0') )>0
        ,1, 0) AS weblog_sql_03_comb,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '),  'waitfor(.*?)delay') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '),  'db(.*?)sql(.*?)server') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '),  'cast(.*?)chr') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'cast(.*?)chr[\\%]') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'cast(.*?)chr[\\(]') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'cast(.*?)char[\\%]') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'cast(.*?)char[\\(]') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'upper(.*?)xmltype') )>0
        ,1, 0) AS weblog_sql_04_comb,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '),  'sql(.*?)select') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '),  'query(.*?)select') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), '[\\=]yes') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), '[\\=]true') )>0
        ,1, 0) AS weblog_sql_05_comb,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'script(.*?)alert') )>0
        OR INT(RLIKE(LOWER(web_log), 'onmouseover') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'eval[\\/][\\*]') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'eval[\\%]') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'eval[\\|]') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'eval[\\-]') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'eval[\\(]') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'onerror(.*?)alert') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'document(.*?)open') )>0
        ,1, 0) AS weblog_xss_01_comb,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '),  'wget[\\+](.*?)ttp') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'wget[\\-]c(.*?)ttp') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'wget[\\%]20(.*?)ttp') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'wget (.*?)ttp') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'wget[\\$][\\{](.*?)ttp') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '),  'wget[\\%](.*?)ttp') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '),  'wget[\\$](.*?)ttp') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'rm(.*?)[\\-]rf') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'cd (.*?)tmp') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'cd[\\%]20(.*?)tmp') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'cd[\\+](.*?)tmp') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'cd[\\$][\\{](.*?)tmp') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '),  'cd(.*?)[\\/]tmp') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '),  'chmod[\\+](.*?)777') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '),  'chmod[\\%](.*?)777') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'chmod[\\$](.*?)777') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'chmod[\\(][\\$](.*?)777') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'chmod[\\+](.*?)777') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'chmod (.*?)777') )>0
        ,1, 0) AS weblog_cmd_01_comb,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'syscmd(.*?)cmd') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '),  'exec(.*?)cmd(.*?)dir') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '),  'exec(.*?)cmd(.*?)ls') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'cmd(.*?)open') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'get(.*?)[\\%]20ping') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'get(.*?)[\\%]27ping') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'get(.*?)[\\=]ping') )>0
        ,1, 0) AS weblog_cmd_02_comb,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'command') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '),  'ping[\\%]20') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '),  'ping[\\+]') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '),  'echo[\\%]20') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '),  'echo[\\+]') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '),  'cat[\\%]20') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '),   'cat[\\+]') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'shell[\\_]exe') )>0
        ,1, 0) AS weblog_cmd_03_comb,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'current[\\_]config(.*?)passwd') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'well(.*?)known') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'backup(.*?)sql') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'robots(.*?)txt') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'etc(.*?)passwd') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'etc(.*?)shadow') )>0
            OR INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'bash(.*?)history') )>0
        ,1, 0) AS weblog_dir_01_comb,

        (SIZE(SPLIT(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), '[\\.][\\.][\\/]')) -1)
        + (SIZE(SPLIT(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), '[\\.][\\.][\\%]2f')) -1)
        + (SIZE(SPLIT(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), '[\\%]2e[\\%]2e[\\%]2f')) -1)
                AS weblog_dir_count,

        SIZE(SPLIT(IF(ISNULL(web_log), '', web_log), '[\\&]'))-1 AS weblog_ampersand_count,

        SIZE(SPLIT(IF(ISNULL(web_log), '', web_log), '[\\;]'))-1 AS weblog_semicolon_count

    FROM table
    """



# ips_query의 'AS ' 부터 ',' 또는 ' ' 까지 추출하여 리스트 생성
ips_feature_list = re.findall(r'AS (.*?)[\,|\s]', ips_query)
# ips_feature_lsit 의 'ips_' 제거
ips_feature_list = [x.replace('ips_', '') for x in ips_feature_list]
# ips_feature_list에서 'whitelist' 를 포함하는 element 제거
ips_attack_feature_list = [x for x in ips_feature_list if 'whitelist' not in x]
ips_whitelist_feature_list = [x for x in ips_feature_list if 'whitelist' in x]
ips_whitelist_feature = ips_whitelist_feature_list[0]
'''연속형 피처 - 특수문자 개수 (&, ;), dir_count'''
ips_attack_conti_feature_list = [x for x in ips_feature_list if 'count' in x]
ips_attack_dir_conti_feature, ips_attack_and_conti_feature, ips_attack_semico_conti_feature = ips_attack_conti_feature_list[:3]

# waf_query의 'AS ' 부터 ',' 또는 ' ' 까지 추출하여 리스트 생성
waf_feature_list = re.findall(r'AS (.*?)[\,|\s]', waf_query)
# waf_feature_lsit 의 'waf_' 제거
waf_feature_list = [x.replace('waf_', '') for x in waf_feature_list]
'''연속형 피처 - 특수문자 개수 (&, ;), dir_count'''
waf_attack_conti_feature_list = [x for x in waf_feature_list if 'count' in x]
waf_attack_dir_conti_feature, waf_attack_and_conti_feature, waf_attack_semico_conti_feature = waf_attack_conti_feature_list[:3]
  
# web_query의 'AS ' 부터 ',' 또는 ' ' 까지 추출하여 리스트 생성
web_feature_list = re.findall(r'AS (.*?)[\,|\s]', web_query)
'''연속형 피처 - 특수문자 개수 (&, ;), dir_count'''
web_attack_conti_feature_list = [x for x in web_feature_list if 'count' in x]
web_attack_dir_conti_feature, web_attack_and_conti_feature, web_attack_semico_conti_feature = web_attack_conti_feature_list[:3]



# ips_query '\\n|\\r|\\t', 'http/1.' 는 제거, 단 regex = False
ips_attack_query = ips_query.replace('\\n|\\r|\\t', '').replace('http/1.', '')
# ips_attack_query 에서 ips_payload_whitelist  이후 추출
ips_attack_query = ips_attack_query.split('ips_payload_whitelist')[1]

# ips_attack_query의 '' 안에 있는 문자열들을 추출하여 리스트 생성, 
ips_ai_field = re.findall(r'\'(.*?)\'', ips_attack_query)

# ips_ai_field에서 'remove_string' 는 제거
ips_ai_field = [x for x in ips_ai_field if x != '' and x != ' ']

# ips_attack_new_sql_query 에서 'AS' 를 기준으로 분할
ips_attack_new_sql_query_split = ips_attack_query.split('AS')
ips_auth_field, ips_bof_field, ips_cmd_1_field ,ips_cmd_2_field, ips_code_field, ips_dir_1_field, ips_dir_2_field, ips_dir_count_field, ips_cgi_field, ips_wp_field, ips_error_field, ips_file_field, ips_http_method_field, ips_malware_field, ips_rce_field, ips_sql_1_field, ips_sql_2_field, ips_useragent_field, ips_php_field, ips_xss_field, ips_and_count_field, ips_semico_count_field = ips_attack_new_sql_query_split[:22]

ips_auth_field, ips_bof_field, ips_cmd_1_field ,ips_cmd_2_field, ips_code_field, ips_dir_1_field, ips_dir_2_field, ips_dir_count_field, ips_cgi_field, ips_wp_field, ips_error_field, ips_file_field, ips_http_method_field, ips_malware_field, ips_rce_field, ips_sql_1_field, ips_sql_2_field, ips_useragent_field, ips_php_field, ips_xss_field, ips_and_count_field, ips_semico_count_field = list(map(lambda x: re.findall(r'\'(.*?)\'', x), 
                                                                        [ips_auth_field, ips_bof_field, ips_cmd_1_field ,ips_cmd_2_field, ips_code_field, ips_dir_1_field, ips_dir_2_field, ips_dir_count_field, ips_cgi_field, ips_wp_field, ips_error_field, ips_file_field, ips_http_method_field, ips_malware_field, ips_rce_field, ips_sql_1_field, ips_sql_2_field, ips_useragent_field, ips_php_field, ips_xss_field, ips_and_count_field, ips_semico_count_field]))
ips_auth_field, ips_bof_field, ips_cmd_1_field ,ips_cmd_2_field, ips_code_field, ips_dir_1_field, ips_dir_2_field, ips_dir_count_field, ips_cgi_field, ips_wp_field, ips_error_field, ips_file_field, ips_http_method_field, ips_malware_field, ips_rce_field, ips_sql_1_field, ips_sql_2_field, ips_useragent_field, ips_php_field, ips_xss_field, ips_and_count_field, ips_semico_count_field = list(map(lambda x: [y for y in x if y != '' and y != ' '],
                                                                        [ips_auth_field, ips_bof_field, ips_cmd_1_field ,ips_cmd_2_field, ips_code_field, ips_dir_1_field, ips_dir_2_field, ips_dir_count_field, ips_cgi_field, ips_wp_field, ips_error_field, ips_file_field, ips_http_method_field, ips_malware_field, ips_rce_field, ips_sql_1_field, ips_sql_2_field, ips_useragent_field, ips_php_field, ips_xss_field, ips_and_count_field, ips_semico_count_field])) 

ips_field_feature_dict = {field: feature for field, feature in zip(
    [ips_auth_field, ips_bof_field, ips_cmd_1_field ,ips_cmd_2_field, ips_code_field, ips_dir_1_field, ips_dir_2_field, ips_dir_count_field, ips_cgi_field, ips_wp_field, ips_error_field, ips_file_field, ips_http_method_field, ips_malware_field, ips_rce_field, ips_sql_1_field, ips_sql_2_field, ips_useragent_field, ips_php_field, ips_xss_field, ips_and_count_field, ips_semico_count_field],
    ips_attack_feature_list
    )}


# waf_query '\\n|\\r|\\t', 'http/1.' 는 제거, 단 regex = False
waf_attack_query = waf_query.replace('\\n|\\r|\\t', '').replace('http/1.', '')

# waf_attack_query의 '' 안에 있는 문자열들을 추출하여 리스트 생성, 
waf_ai_field = re.findall(r'\'(.*?)\'', waf_attack_query)

# waf_ai_field에서 'remove_string' 는 제거
waf_ai_field = [x for x in waf_ai_field if x != '' and x != ' ']

# waf_attack_new_sql_query 에서 'AS' 를 기준으로 분할
waf_attack_new_sql_query_split = waf_attack_query.split('AS')
waf_auth_field, waf_bof_field, waf_cmd_1_field ,waf_cmd_2_field, waf_code_field, waf_dir_1_field, waf_dir_2_field, waf_dir_count_field, waf_cgi_field, waf_wp_field, waf_error_field, waf_file_field, waf_http_method_field, waf_malware_field, waf_rce_field, waf_sql_1_field, waf_sql_2_field, waf_useragent_field, waf_php_field, waf_xss_field, waf_and_count_field, waf_semico_count_field = waf_attack_new_sql_query_split[:22]

waf_auth_field, waf_bof_field, waf_cmd_1_field ,waf_cmd_2_field, waf_code_field, waf_dir_1_field, waf_dir_2_field, waf_dir_count_field, waf_cgi_field, waf_wp_field, waf_error_field, waf_file_field, waf_http_method_field, waf_malware_field, waf_rce_field, waf_sql_1_field, waf_sql_2_field, waf_useragent_field, waf_php_field, waf_xss_field, waf_and_count_field, waf_semico_count_field = list(map(lambda x: re.findall(r'\'(.*?)\'', x), 
                                                                        [waf_auth_field, waf_bof_field, waf_cmd_1_field ,waf_cmd_2_field, waf_code_field, waf_dir_1_field, waf_dir_2_field, waf_dir_count_field, waf_cgi_field, waf_wp_field, waf_error_field, waf_file_field, waf_http_method_field, waf_malware_field, waf_rce_field, waf_sql_1_field, waf_sql_2_field, waf_useragent_field, waf_php_field, waf_xss_field, waf_and_count_field, waf_semico_count_field]))
waf_auth_field, waf_bof_field, waf_cmd_1_field ,waf_cmd_2_field, waf_code_field, waf_dir_1_field, waf_dir_2_field, waf_dir_count_field, waf_cgi_field, waf_wp_field, waf_error_field, waf_file_field, waf_http_method_field, waf_malware_field, waf_rce_field, waf_sql_1_field, waf_sql_2_field, waf_useragent_field, waf_php_field, waf_xss_field, waf_and_count_field, waf_semico_count_field = list(map(lambda x: [y for y in x if y != '' and y != ' '],
                                                                        [waf_auth_field, waf_bof_field, waf_cmd_1_field ,waf_cmd_2_field, waf_code_field, waf_dir_1_field, waf_dir_2_field, waf_dir_count_field, waf_cgi_field, waf_wp_field, waf_error_field, waf_file_field, waf_http_method_field, waf_malware_field, waf_rce_field, waf_sql_1_field, waf_sql_2_field, waf_useragent_field, waf_php_field, waf_xss_field, waf_and_count_field, waf_semico_count_field]))  

waf_field_feature_dict = {field: feature for field, feature in zip(
    [waf_auth_field, waf_bof_field, waf_cmd_1_field ,waf_cmd_2_field, waf_code_field, waf_dir_1_field, waf_dir_2_field, waf_dir_count_field, waf_cgi_field, waf_wp_field, waf_error_field, waf_file_field, waf_http_method_field, waf_malware_field, waf_rce_field, waf_sql_1_field, waf_sql_2_field, waf_useragent_field, waf_php_field, waf_xss_field, waf_and_count_field, waf_semico_count_field],
    waf_feature_list
    )}


# web_query '\\n|\\r|\\t', 'http/1.' 는 제거, 단 regex = False
web_attack_query = web_query.replace('\\n|\\r|\\t', '').replace('http/1.', '')
# web_attack_query의 '' 안에 있는 문자열들을 추출하여 리스트 생성, 
web_ai_field = re.findall(r'\'(.*?)\'', web_attack_query)
# web_ai_field에서 'remove_string' 는 제거
web_ai_field = [x for x in web_ai_field if x != '' and x != ' ']

# web_attack_new_sql_query_split 에서 'AS' 를 기준으로 분할
web_attack_new_sql_query_split = web_attack_query.split('AS')
web_sql_1_field, web_sql_2_field, web_sql_3_field, web_sql_4_field, web_sql_5_field, web_xss_field, web_cmd_1_field, web_cmd_2_field, web_cmd_3_field, web_dir_1_field, web_dir_count_field, web_and_count_field, web_semico_count_field = web_attack_new_sql_query_split[:13]
web_sql_1_field, web_sql_2_field, web_sql_3_field, web_sql_4_field, web_sql_5_field, web_xss_field, web_cmd_1_field, web_cmd_2_field, web_cmd_3_field, web_dir_1_field, web_dir_count_field, web_and_count_field, web_semico_count_field = list(map(lambda x: re.findall(r'\'(.*?)\'', x), 
                                                                        [web_sql_1_field, web_sql_2_field, web_sql_3_field, web_sql_4_field, web_sql_5_field, web_xss_field, web_cmd_1_field, web_cmd_2_field, web_cmd_3_field, web_dir_1_field, web_dir_count_field, web_and_count_field, web_semico_count_field]))
web_sql_1_field, web_sql_2_field, web_sql_3_field, web_sql_4_field, web_sql_5_field, web_xss_field, web_cmd_1_field, web_cmd_2_field, web_cmd_3_field, web_dir_1_field, web_dir_count_field, web_and_count_field, web_semico_count_field = list(map(lambda x: [y for y in x if y != '' and y != ' '],
                                                                        [web_sql_1_field, web_sql_2_field, web_sql_3_field, web_sql_4_field, web_sql_5_field, web_xss_field, web_cmd_1_field, web_cmd_2_field, web_cmd_3_field, web_dir_1_field, web_dir_count_field, web_and_count_field, web_semico_count_field])) 

web_field_feature_dict = {field: feature for field, feature in zip(
    [web_sql_1_field, web_sql_2_field, web_sql_3_field, web_sql_4_field, web_sql_5_field, web_xss_field, web_cmd_1_field, web_cmd_2_field, web_cmd_3_field, web_dir_1_field, web_dir_count_field, web_and_count_field, web_semico_count_field],
    web_feature_list
    )}







