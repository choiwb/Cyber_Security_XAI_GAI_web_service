


# import psycopg2 as pg2
# import os
# import pandas as pd
# import torch
# from transformers import AutoTokenizer, pipeline

import re
import pickle
import itertools
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

SERVER_IP = '0.0.0.0'
PORT = 17171


# 2023/04/27 IPS 모델 - Light GBM
new_IPS_model_path = 'save_model/DSS_IPS_LGB_20230427.pkl'
# 위 모델을, SHAP의 TreeExplainer 연산 및 저장
IPS_explainer_path = 'save_model/DSS_IPS_LGB_explainer_20230427.pkl'

IPS_model = pickle.load(open(new_IPS_model_path, 'rb'))


# 2023/04/27 WAF 모델 - Light GBM
new_WAF_model_path = 'save_model/DSS_WAF_LGB_20230427.pkl'
# 위 모델을, SHAP의 TreeExplainer 연산 및 저장
WAF_explainer_path = 'save_model/DSS_WAF_LGB_explainer_20230427.pkl'

WAF_model = pickle.load(open(new_WAF_model_path, 'rb'))


# 2023/04/04 WEB 모델 - Light GBM
WEB_model_path = 'save_model/DSS_WEB_LGB_20230404.pkl'
# 위 모델을, SHAP의 TreeExplainer 연산 및 저장
WEB_explainer_path = 'save_model/DSS_WEB_LGB_explainer_20230404.pkl'

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

# sep_list = [' ', ',', ';', ':', '-', '/', '_']
# sep_list = [' ', 'mozilla', 'windows', 'gecko', 'like', 'applewebkit', 'chrome', 'safari', 'khtml']
sep_list = [' ']

sep_str = '|'.join(sep_list)

vectorizer = CountVectorizer(lowercase = True,
                             tokenizer = lambda x: re.split(sep_str, x),
                              vocabulary = tfidf_word)

# GeoIP2의 국가명 조회 DB 경로 - 20230419 업데이트 기준
geoip_country_db_path = 'save_model/GeoLite2_Country_20230419.mmdb'


ips_query = """
    
    SELECT
   
        IF((SIZE(SPLIT(REGEXP_REPLACE(payload, '\\n|\\r|\\t', ' '), 'GET(.*?)HTTP/1.')) -1)
        + (SIZE(SPLIT(REGEXP_REPLACE(payload, '\\n|\\r|\\t', ' '), 'POST(.*?)HTTP/1.')) -1)
        + (SIZE(SPLIT(REGEXP_REPLACE(payload, '\\n|\\r|\\t', ' '), 'HEAD(.*?)HTTP/1.')) -1)
        + (SIZE(SPLIT(REGEXP_REPLACE(payload, '\\n|\\r|\\t', ' '), 'OPTION(.*?)HTTP/1.')) -1) >= 2
        , 0, 1) AS ips_payload_whitelist,

        IF((INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'select(.*?)from') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'select(.*?)count') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'select(.*?)distinct') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'union(.*?)select') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'select(.*?)extractvalue(.*?)xmltype') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'and(.*?)select') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'from(.*?)generate(.*?)series') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'from(.*?)group(.*?)by') )>0)
                ,1, 0) AS ips_payload_sql_comb_01,

        IF((INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'case(.*?)when') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'then(.*?)else') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'like') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'sleep') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'delete') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'drop') )>0)
                ,1, 0) AS ips_payload_sql_comb_02,

        IF((INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'waitfor(.*?)delay') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'db(.*?)sql(.*?)server') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'cast(.*?)chr') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'upper(.*?)xmltype') )>0)
                ,1, 0) AS ips_payload_sql_comb_03,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'script(.*?)alert') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'onerror(.*?)alert') )>0
        OR INT(RLIKE(LOWER(payload), 'eval') )>0
                    ,1, 0) AS ips_payload_xss_comb_01,

        IF((INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'wget(.*?)ttp') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'chmod(.*?)777') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'rm(.*?)-rf') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'cd(.*?)tmp') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'getruntime(.*?)exec') )>0)
                ,1, 0) AS ips_payload_cmd_comb_01,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'jndi(.*?)dap') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),'jndi(.*?)dns') )>0
                ,1, 0) AS ips_payload_log4j_comb_01,
        
        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'document(.*?)createelement') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'cgi(.*?)bin') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'document(.*?)forms') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'document(.*?)location') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'fckeditor(.*?)filemanager') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'manager(.*?)html') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'current_config(.*?)passwd') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'currentsetting(.*?)htm') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'well(.*?)known') )>0
                ,1, 0) AS ips_payload_word_comb_01,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'bash(.*?)history') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'apache(.*?)struts') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'document(.*?)open') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'backup(.*?)sql') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'robots(.*?)txt') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'sqlexec(.*?)php') )>0
        OR INT(RLIKE(LOWER(payload), 'htaccess') )>0
        OR INT(RLIKE(LOWER(payload), 'htpasswd') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'cgi(.*?)cgi') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'api(.*?)ping') )>0
                ,1, 0) AS ips_payload_word_comb_02,

        IF(INT(RLIKE(LOWER(payload), 'aaaaaaaaaa') )>0
        OR INT(RLIKE(LOWER(payload), 'cacacacaca') )>0
        OR INT(RLIKE(LOWER(payload), 'mozi[\\.]') )>0
        OR INT(RLIKE(LOWER(payload), 'bingbot') )>0
        OR INT(RLIKE(LOWER(payload), 'md5') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'count(.*?)cgi(.*?)http') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'this(.*?)program(.*?)can') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'get(.*?)ping') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'msadc(.*?)dll(.*?)http') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'filename(.*?)asp') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'filename(.*?)jsp') )>0
        OR INT(RLIKE(LOWER(payload), 'powershell'))>0
        OR INT(RLIKE(LOWER(payload), '[\\.]env'))>0
                ,1, 0) AS ips_payload_word_comb_03,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'cmd(.*?)open') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'echo(.*?)shellshock') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'php(.*?)echo') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'admin(.*?)php') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'script(.*?)setup(.*?)php') )>0
        OR INT(RLIKE(LOWER(payload), 'phpinfo') )>0
        OR INT(RLIKE(LOWER(payload), 'administrator') )>0
        OR INT(RLIKE(LOWER(payload), 'phpmyadmin') )>0
        OR INT(RLIKE(LOWER(payload), 'access') )>0
        OR INT(RLIKE(LOWER(payload), 'mdb') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'wise(.*?)survey(.*?)admin') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'admin(.*?)serv(.*?)admpw') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'php(.*?)create(.*?)function') )>0
                ,1, 0) AS ips_payload_word_comb_04,

        IF(INSTR(LOWER(payload), '/bash') >0
        OR INSTR(LOWER(payload), '.cgi')>0
        OR INSTR(LOWER(payload), '/test')>0
        OR INSTR(LOWER(payload), 'get /hnap1')>0
        OR INSTR(LOWER(payload), 'jdatabasedrivermysqli')>0
        OR INSTR(LOWER(payload), '/a.jsp')>0
        OR INSTR(LOWER(payload), '/tcsh')>0
        OR INSTR(LOWER(payload), '.asp;.jpg')>0
        OR INSTR(LOWER(payload), 'delete /')>0
        OR INSTR(LOWER(payload), 'put /')>0
                    ,1, 0) AS ips_payload_word_comb_05,

        IF(INT(RLIKE(LOWER(payload), '80040e07(.*?)font') )>0
        OR INT(RLIKE(LOWER(payload), '80040e14(.*?)font') )>0
        OR INT(RLIKE(LOWER(payload), 'search(.*?)cgi') )>0
        OR INT(RLIKE(LOWER(payload), 'bbs(.*?)forum(.*?)cgi') )>0
        OR INT(RLIKE(LOWER(payload), 'web(.*?)store(.*?)cgi') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'upload(.*?)asp') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'upload(.*?)php') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'cgi(.*?)cgi(.*?)cgimail(.*?)exe') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'cgi(.*?)cgi(.*?)fpcount(.*?)exe') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'cgi(.*?)cgi(.*?)rguest(.*?)exe') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'count(.*?)cgi(.*?)http') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'php(.?)content(.*?)type') )>0
                    ,1, 0) AS ips_payload_word_comb_06,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'wp-login') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'wp-content') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'wp-include') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'wp-config') )>0
                ,1, 0) AS ips_payload_wp_comb_01,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'etc(.*?)passwd') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'etc(.*?)shadow') )>0
                ,1, 0) AS ips_payload_dir_access_comb_01,

        (SIZE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), '[\\.][\\.]/')) -1)
        + (SIZE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), '[\\.][\\.]%2f')) -1)
        + (SIZE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), '[\\.][\\.]%252f')) -1)
        + (SIZE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), '%2e%2e%2f')) -1)
        + (SIZE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), '%2e%2e%25%32%66')) -1)
        AS ips_payload_dir_access_comb_02,

        IF((INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'user(.*?)agent(.*?)zgrab') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'user(.*?)agent(.*?)nmap') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'user(.*?)agent(.*?)dirbuster') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'user(.*?)agent(.*?)ahrefsbot') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'user(.*?)agent(.*?)baiduspider') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'user(.*?)agent(.*?)mj12bot') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'user(.*?)agent(.*?)petalbot') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'user(.*?)agent(.*?)curl/') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'user(.*?)agent(.*?)semrushbot') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'user(.*?)agent(.*?)masscan') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'user(.*?)agent(.*?)sqlmap') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'user(.*?)agent(.*?)urlgrabber(.*?)yum') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'user(.*?)agent(.*?)zmeu') )>0)
                ,1, 0) AS ips_payload_useragent_comb_01


    FROM table
    """

waf_query = """
    
    SELECT
   
        IF((INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'select(.*?)from') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'select(.*?)count') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'select(.*?)distinct') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'union(.*?)select') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'select(.*?)extractvalue(.*?)xmltype') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'and(.*?)select') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'from(.*?)generate(.*?)series') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'from(.*?)group(.*?)by') )>0)
                ,1, 0) AS waf_payload_sql_comb_01,

        IF((INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'case(.*?)when') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'then(.*?)else') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'like') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'sleep') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'delete') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'drop') )>0)
                ,1, 0) AS waf_payload_sql_comb_02,

        IF((INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'waitfor(.*?)delay') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'db(.*?)sql(.*?)server') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'cast(.*?)chr') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'upper(.*?)xmltype') )>0)
                ,1, 0) AS waf_payload_sql_comb_03,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'script(.*?)alert') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'onerror(.*?)alert') )>0
        OR INT(RLIKE(LOWER(payload), 'eval') )>0
                    ,1, 0) AS waf_payload_xss_comb_01,

        IF((INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'wget(.*?)ttp') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'chmod(.*?)777') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'rm(.*?)-rf') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'cd(.*?)tmp') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'getruntime(.*?)exec') )>0)
                ,1, 0) AS waf_payload_cmd_comb_01,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'jndi(.*?)dap') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),'jndi(.*?)dns') )>0
                ,1, 0) AS waf_payload_log4j_comb_01,
        
        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'document(.*?)createelement') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'cgi(.*?)bin') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'document(.*?)forms') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'document(.*?)location') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'fckeditor(.*?)filemanager') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'manager(.*?)html') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'current_config(.*?)passwd') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'currentsetting(.*?)htm') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'well(.*?)known') )>0
                ,1, 0) AS waf_payload_word_comb_01,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'bash(.*?)history') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'apache(.*?)struts') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'document(.*?)open') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'backup(.*?)sql') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'robots(.*?)txt') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'sqlexec(.*?)php') )>0
        OR INT(RLIKE(LOWER(payload), 'htaccess') )>0
        OR INT(RLIKE(LOWER(payload), 'htpasswd') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'cgi(.*?)cgi') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'api(.*?)ping') )>0
                ,1, 0) AS waf_payload_word_comb_02,

        IF(INT(RLIKE(LOWER(payload), 'aaaaaaaaaa') )>0
        OR INT(RLIKE(LOWER(payload), 'cacacacaca') )>0
        OR INT(RLIKE(LOWER(payload), 'mozi[\\.]') )>0
        OR INT(RLIKE(LOWER(payload), 'bingbot') )>0
        OR INT(RLIKE(LOWER(payload), 'md5') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'count(.*?)cgi(.*?)http') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'this(.*?)program(.*?)can') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'get(.*?)ping') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'msadc(.*?)dll(.*?)http') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'filename(.*?)asp') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'filename(.*?)jsp') )>0
        OR INT(RLIKE(LOWER(payload), 'powershell'))>0
        OR INT(RLIKE(LOWER(payload), '[\\.]env'))>0
                ,1, 0) AS waf_payload_word_comb_03,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'cmd(.*?)open') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'echo(.*?)shellshock') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'php(.*?)echo') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'admin(.*?)php') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'script(.*?)setup(.*?)php') )>0
        OR INT(RLIKE(LOWER(payload), 'phpinfo') )>0
        OR INT(RLIKE(LOWER(payload), 'administrator') )>0
        OR INT(RLIKE(LOWER(payload), 'phpmyadmin') )>0
        OR INT(RLIKE(LOWER(payload), 'access') )>0
        OR INT(RLIKE(LOWER(payload), 'mdb') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'wise(.*?)survey(.*?)admin') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'admin(.*?)serv(.*?)admpw') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'php(.*?)create(.*?)function') )>0
                ,1, 0) AS waf_payload_word_comb_04,

        IF(INSTR(LOWER(payload), '/bash') >0
        OR INSTR(LOWER(payload), '.cgi')>0
        OR INSTR(LOWER(payload), '/test')>0
        OR INSTR(LOWER(payload), 'get /hnap1')>0
        OR INSTR(LOWER(payload), 'jdatabasedrivermysqli')>0
        OR INSTR(LOWER(payload), '/a.jsp')>0
        OR INSTR(LOWER(payload), '/tcsh')>0
        OR INSTR(LOWER(payload), '.asp;.jpg')>0
        OR INSTR(LOWER(payload), 'delete /')>0
        OR INSTR(LOWER(payload), 'put /')>0
                    ,1, 0) AS waf_payload_word_comb_05,

        IF(INT(RLIKE(LOWER(payload), '80040e07(.*?)font') )>0
        OR INT(RLIKE(LOWER(payload), '80040e14(.*?)font') )>0
        OR INT(RLIKE(LOWER(payload), 'search(.*?)cgi') )>0
        OR INT(RLIKE(LOWER(payload), 'bbs(.*?)forum(.*?)cgi') )>0
        OR INT(RLIKE(LOWER(payload), 'web(.*?)store(.*?)cgi') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'upload(.*?)asp') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'upload(.*?)php') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'cgi(.*?)cgi(.*?)cgimail(.*?)exe') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'cgi(.*?)cgi(.*?)fpcount(.*?)exe') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'cgi(.*?)cgi(.*?)rguest(.*?)exe') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'count(.*?)cgi(.*?)http') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'php(.?)content(.*?)type') )>0
                    ,1, 0) AS waf_payload_word_comb_06,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'wp-login') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'wp-content') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'wp-include') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'wp-config') )>0
                ,1, 0) AS waf_payload_wp_comb_01,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'etc(.*?)passwd') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'etc(.*?)shadow') )>0
                ,1, 0) AS waf_payload_dir_access_comb_01,

        (SIZE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), '[\\.][\\.]/')) -1)
        + (SIZE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), '[\\.][\\.]%2f')) -1)
        + (SIZE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), '[\\.][\\.]%252f')) -1)
        + (SIZE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), '%2e%2e%2f')) -1)
        + (SIZE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), '%2e%2e%25%32%66')) -1)
        AS waf_payload_dir_access_comb_02,

        IF((INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'user(.*?)agent(.*?)zgrab') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'user(.*?)agent(.*?)nmap') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'user(.*?)agent(.*?)dirbuster') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'user(.*?)agent(.*?)ahrefsbot') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'user(.*?)agent(.*?)baiduspider') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'user(.*?)agent(.*?)mj12bot') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'user(.*?)agent(.*?)petalbot') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'user(.*?)agent(.*?)curl/') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'user(.*?)agent(.*?)semrushbot') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'user(.*?)agent(.*?)masscan') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'user(.*?)agent(.*?)sqlmap') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'user(.*?)agent(.*?)urlgrabber(.*?)yum') )>0)
        OR (INSTR(LOWER(payload),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[1],  'user(.*?)agent(.*?)zmeu') )>0)
                ,1, 0) AS waf_payload_useragent_comb_01
        
    FROM table
    """




web_query = """

SELECT 

        IF((INSTR(LOWER(web_log),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'select(.*?)from') )>0)
        OR (INSTR(LOWER(web_log),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'select(.*?)count') )>0)
        OR (INSTR(LOWER(web_log),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'select(.*?)concat') )>0)
        OR (INSTR(LOWER(web_log),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'select(.*?)distinct') )>0)
        OR (INSTR(LOWER(web_log),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'union(.*?)select') )>0)
        OR (INSTR(LOWER(web_log),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'select(.*?)extractvalue(.*?)xmltype') )>0)
        OR (INSTR(LOWER(web_log),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'from(.*?)generate(.*?)series') )>0)
        OR (INSTR(LOWER(web_log),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'from(.*?)group(.*?)by') )>0)
                ,1, 0) AS weblog_sql_comb_01,

        IF((INSTR(LOWER(web_log),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'case(.*?)when') )>0)
        OR (INSTR(LOWER(web_log),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'then(.*?)else') )>0)
        OR (INSTR(LOWER(web_log),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'like') )>0)
        OR (INSTR(LOWER(web_log),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'sleep') )>0)
        OR (INSTR(LOWER(web_log),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'delete(.*?)from') )>0)
                ,1, 0) AS weblog_sql_comb_02,

        IF((INSTR(LOWER(web_log),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'where_framework') )>0)
        OR (INSTR(LOWER(web_log),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'sql_server') )>0)
        OR (INSTR(LOWER(web_log),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'order=1') )>0)
        OR (INSTR(LOWER(web_log),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'id=0') )>0)
                ,1, 0) AS weblog_sql_comb_03,

        IF((INSTR(LOWER(web_log),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'waitfor(.*?)delay') )>0)
        OR (INSTR(LOWER(web_log),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'db(.*?)sql(.*?)server') )>0)
        OR (INSTR(LOWER(web_log),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'cast(.*?)chr') )>0)
        OR (INSTR(LOWER(web_log),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'upper(.*?)xmltype') )>0)
                ,1, 0) AS weblog_sql_comb_04,

        IF((INSTR(LOWER(web_log),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'sql(.*?)select') )>0)
        OR (INSTR(LOWER(web_log),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'query(.*?)select') )>0)
        OR (INSTR(LOWER(web_log),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  '=yes') )>0)
        OR (INSTR(LOWER(web_log),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  '=true') )>0)
                ,1, 0) AS weblog_sql_comb_05,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'script(.*?)alert') )>0
        OR INT(RLIKE(LOWER(web_log), 'onmouseover') )>0
        OR INT(RLIKE(LOWER(web_log), 'eval') )>0
                ,1, 0) AS weblog_xss_comb_01,

        IF((INSTR(LOWER(web_log),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'wget(.*?)ttp') )>0)
        OR (INSTR(LOWER(web_log),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'chmod(.*?)777') )>0)
        OR (INSTR(LOWER(web_log),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'rm(.*?)-rf') )>0)
        OR (INSTR(LOWER(web_log),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'cd(.*?)tmp') )>0)
                ,1, 0) AS weblog_cmd_comb_01,

        IF((INSTR(LOWER(web_log),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'syscmd(.*?)cmd') )>0)
        OR (INSTR(LOWER(web_log),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'exec(.*?)cmd(.*?)dir') )>0)
        OR (INSTR(LOWER(web_log),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'exec(.*?)cmd(.*?)ls') )>0)
                ,1, 0) AS weblog_cmd_comb_02,

        IF((INSTR(LOWER(web_log),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'command') )>0)
        OR (INSTR(LOWER(web_log),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'ping%20') )>0)
        OR (INSTR(LOWER(web_log),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'ping[\\+]') )>0)
        OR (INSTR(LOWER(web_log),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'echo%20') )>0)
        OR (INSTR(LOWER(web_log),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'echo[\\+]') )>0)
        OR (INSTR(LOWER(web_log),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'cat%20') )>0)
        OR (INSTR(LOWER(web_log),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'cat[\\+]') )>0)
        OR (INSTR(LOWER(web_log),'http/1.') > 0 AND INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  'shell_exe') )>0)
                ,1, 0) AS weblog_cmd_comb_03,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), 'etc(.*?)passwd') )>0
                ,1, 0) AS weblog_dir_access_comb_01,

        (SIZE(SPLIT(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), '[\\.][\\.]/')) -1)
        + (SIZE(SPLIT(REGEXP_REPLACE(LOWER(web_log), '\\n|\\r|\\t', ' '), '[\\.][\\.]%2f')) -1)
                AS weblog_dir_access_comb_02


FROM table

"""



# attack_new_sql_query '\\n|\\r|\\t', 'http/1.' 는 제거, 단 regex = False
attack_query = waf_query.replace('\\n|\\r|\\t', '').replace('http/1.', '')
# new_sql_query의 '' 안에 있는 문자열들을 추출하여 리스트 생성, 
ai_field = re.findall(r'\'(.*?)\'', attack_query)
# ai_field에서 'remove_string' 는 제거
ai_field = [x for x in ai_field if x != '' and x != ' ']


# attack_new_sql_query 에서 'AS' 를 기준으로 분할
attack_new_sql_query_split = attack_query.split('AS')

sql_1, sql_2, sql_3, xss, cmd, log4j, word_1, word_2, word_3, word_4, word_5, word_6, wp, dir_access_1, dir_access_2, user_agent = attack_new_sql_query_split[:16]
sql_1, sql_2, sql_3, xss, cmd, log4j, word_1, word_2, word_3, word_4, word_5, word_6, wp, dir_access_1, dir_access_2, user_agent = list(map(lambda x: re.findall(r'\'(.*?)\'', x), 
                                                                        [sql_1, sql_2, sql_3, xss, cmd, log4j, word_1, word_2, word_3, word_4, word_5, word_6, wp, dir_access_1, dir_access_2, user_agent]))
sql_1, sql_2, sql_3, xss, cmd, log4j, word_1, word_2, word_3, word_4, word_5, word_6, wp, dir_access_1, dir_access_2, user_agent = list(map(lambda x: [y for y in x if y != '' and y != ' '],
                                                                        [sql_1, sql_2, sql_3, xss, cmd, log4j, word_1, word_2, word_3, word_4, word_5, word_6, wp, dir_access_1, dir_access_2, user_agent])) 


# attack_new_sql_query '\\n|\\r|\\t', 'http/1.' 는 제거, 단 regex = False
web_attack_query = web_query.replace('\\n|\\r|\\t', '').replace('http/1.', '')
# new_sql_query의 '' 안에 있는 문자열들을 추출하여 리스트 생성, 
web_ai_field = re.findall(r'\'(.*?)\'', web_attack_query)
# ai_field에서 'remove_string' 는 제거
web_ai_field = [x for x in web_ai_field if x != '' and x != ' ']

# web_attack_new_sql_query_split 에서 'AS' 를 기준으로 분할
web_attack_new_sql_query_split = web_attack_query.split('AS')

web_sql_1, web_sql_2, web_sql_3, web_sql_4, web_sql_5, web_xss, web_cmd_1, web_cmd_2, web_cmd_3, web_dir_access_1, web_dir_access_2 = web_attack_new_sql_query_split[:11]
web_sql_1, web_sql_2, web_sql_3, web_sql_4, web_sql_5, web_xss, web_cmd_1, web_cmd_2, web_cmd_3, web_dir_access_1, web_dir_access_2 = list(map(lambda x: re.findall(r'\'(.*?)\'', x), 
                                                                        [web_sql_1, web_sql_2, web_sql_3, web_sql_4, web_sql_5, web_xss, web_cmd_1, web_cmd_2, web_cmd_3, web_dir_access_1, web_dir_access_2]))
web_sql_1, web_sql_2, web_sql_3, web_sql_4, web_sql_5, web_xss, web_cmd_1, web_cmd_2, web_cmd_3, web_dir_access_1, web_dir_access_2 = list(map(lambda x: [y for y in x if y != '' and y != ' '],
                                                                        [web_sql_1, web_sql_2, web_sql_3, web_sql_4, web_sql_5, web_xss, web_cmd_1, web_cmd_2, web_cmd_3, web_dir_access_1, web_dir_access_2])) 









