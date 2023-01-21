

import pickle
import psycopg2 as pg2
import torch
import numpy as np
import scipy as sp
from transformers import AutoTokenizer
from transformers import pipeline, AutoTokenizer
import shap


SERVER_IP = 'SERVER IP'
PORT = 17171 # local


# 절대 경로
IPS_text_model_path = 'MODEL DIR !!!!!!!'
IPS_total_model_path =  'MODEL DIR !!!!!!!'

IPS_text_explainer_path = 'MODEL DIR !!!!!!!'
IPS_total_explainer_path = 'MODEL DIR !!!!!!!'

IPS_pytorch_bert_model_path = 'MODEL DIR !!!!!!!'
# IPS_pytorch_bert_explainer_path = 'MODEL DIR !!!!!!!'


IPS_text_model = pickle.load(open(IPS_text_model_path, 'rb'))
IPS_total_model = pickle.load(open(IPS_total_model_path, 'rb'))

# IPS_text_explainer = pickle.load(open(IPS_text_explainer_path, 'rb'))

# 학습데이터 경로
ips_training_data = 'TRAINING DATA DIR !!!!!'

new_sql_query = """
    
    SELECT
   
        IF(INT(RLIKE(payload, 'VCAvY2dpLWJpbi9waHA0') )>0
        OR INT(RLIKE(payload, 'L2NnaS1iaW4v') )>0
        OR INT(RLIKE(payload, 'IC9jZ2ktYmlu') )>0
        OR INT(RLIKE(payload, 'UE9TVCAvY2dpLWJpbi9waHA/') )>0
        OR INT(RLIKE(payload, 'VCAvY2dpLWJpbi9w') )>0
        OR INT(RLIKE(payload, 'ZGllKEBtZDU=') )>0
        OR INT(RLIKE(payload, 'L2FueWZvcm0yL3VwZGF0ZS9hbnlmb3JtMi5pbmk=') )>0
        OR INT(RLIKE(payload, 'Ly5iYXNoX2hpc3Rvcnk=') )>0
        OR INT(RLIKE(payload, 'L2V0Yy9wYXNzd2Q=') )>0
        OR INT(RLIKE(payload, 'QUFBQUFBQUFBQQ==') )>0
        OR INT(RLIKE(payload, 'IG1hc3NjYW4vMS4w') )>0
        OR INT(RLIKE(payload, 'd2dldA==') )>0
        OR INT(RLIKE(payload, 'MjB3YWl0Zm9yJTIwZGVsYXklMjAn') )>0
        OR INT(RLIKE(payload, 'V0FJVEZPUiBERUxBWQ==') )>0
        OR INT(RLIKE(payload, 'ZXhlYw==') )>0
        OR INT(RLIKE(payload, 'Tm9uZQ==') )>0
        OR INT(RLIKE(payload, 'OyB3Z2V0') )>0
        OR INT(RLIKE(payload, 'VXNlci1BZ2VudDogRGlyQnVzdGVy') )>0
        OR INT(RLIKE(payload, 'cGhwIGRpZShAbWQ1') )>0
        OR INT(RLIKE(payload, 'JTI4U0VMRUNUJTIw') )>0
                ,1, 0) AS ips_00001_payload_base64,

        IF(INT(RLIKE(LOWER(payload), 'select(.*?)from') )>0
        OR INT(RLIKE(LOWER(payload), 'select(.*?)count') )>0
        OR INT(RLIKE(LOWER(payload), 'select(.*?)distinct') )>0
        OR INT(RLIKE(LOWER(payload), 'union(.*?)select') )>0
        OR INT(RLIKE(LOWER(payload), 'select(.*?)extractvalue(.*?)xmltype') )>0
        OR INT(RLIKE(LOWER(payload), 'from(.*?)generate(.*?)series') )>0
        OR INT(RLIKE(LOWER(payload), 'from(.*?)group(.*?)by') )>0
                ,1, 0) AS ips_00001_payload_sql_comb_01,

        IF(INT(RLIKE(LOWER(payload), 'case(.*?)when') )>0
        OR INT(RLIKE(LOWER(payload), 'then(.*?)else') )>0
                ,1, 0) AS ips_00001_payload_sql_comb_02,

        IF(INT(RLIKE(LOWER(payload), 'waitfor(.*?)delay') )>0
        OR INT(RLIKE(LOWER(payload), 'db(.*?)sql(.*?)server') )>0
        OR INT(RLIKE(LOWER(payload), 'cast(.*?)chr') )>0
        OR INT(RLIKE(LOWER(payload), 'like(.*?)http/1.') )>0
        OR INT(RLIKE(LOWER(payload), 'upper(.*?)xmltype') )>0
                ,1, 0) AS ips_00001_payload_sql_comb_03,

        IF(INT(RLIKE(LOWER(payload), 'script(.*?)alert') )>0
                ,1, 0) AS ips_00001_payload_xss_comb_01,

        IF(INT(RLIKE(LOWER(payload), 'wget(.*?)ttp') )>0
        OR INT(RLIKE(LOWER(payload), 'chmod(.*?)777') )>0
        OR INT(RLIKE(LOWER(payload), 'rm(.*?)rf') )>0
        OR INT(RLIKE(LOWER(payload), 'cd(.*?)tmp') )>0
                ,1, 0) AS ips_00001_payload_cmd_comb_01,

        IF(INT(RLIKE(LOWER(payload), 'jndi(.*?)dap') )>0
        OR INT(RLIKE(LOWER(payload),'jndi(.*?)dns') )>0
                ,1, 0) AS ips_00001_payload_log4j_comb_01,

        IF(INT(RLIKE(LOWER(payload), 'etc(.*?)passwd') )>0
        OR INT(RLIKE(LOWER(payload), 'document(.*?)createelement') )>0
        OR INT(RLIKE(LOWER(payload), 'cgi(.*?)bin') )>0
        OR INT(RLIKE(LOWER(payload), 'document(.*?)forms') )>0
        OR INT(RLIKE(LOWER(payload), 'document(.*?)location') )>0
        OR INT(RLIKE(LOWER(payload), 'fckeditor(.*?)filemanager') )>0
        OR INT(RLIKE(LOWER(payload), 'manager(.*?)html') )>0
        OR INT(RLIKE(LOWER(payload), 'current_config(.*?)passwd') )>0
        OR INT(RLIKE(LOWER(payload), 'currentsetting(.*?)htm') )>0
        OR INT(RLIKE(LOWER(payload), 'well(.*?)known') )>0
                ,1, 0) AS ips_00001_payload_word_comb_01,

        IF(INT(RLIKE(LOWER(payload), 'bash(.*?)history') )>0
        OR INT(RLIKE(LOWER(payload), 'apache(.*?)struts') )>0
        OR INT(RLIKE(LOWER(payload), 'document(.*?)open') )>0
        OR INT(RLIKE(LOWER(payload), 'backup(.*?)sql') )>0
        OR INT(RLIKE(LOWER(payload), 'robots(.*?)txt') )>0
        OR INT(RLIKE(LOWER(payload), 'sqlexec(.*?)php') )>0
        OR INT(RLIKE(LOWER(payload), 'exec') )>0
        OR INT(RLIKE(LOWER(payload), 'htaccess') )>0
        OR INT(RLIKE(LOWER(payload), 'htpasswd') )>0
        OR INT(RLIKE(LOWER(payload), 'cgi(.*?)cgi') )>0
        OR INT(RLIKE(LOWER(payload), 'api(.*?)ping') )>0
                ,1, 0) AS ips_00001_payload_word_comb_02,

        IF(INT(RLIKE(LOWER(payload), 'aaaaaaaaaa') )>0
        OR INT(RLIKE(LOWER(payload), 'cacacacaca') )>0
        OR INT(RLIKE(LOWER(payload), 'mozi[\\.]') )>0
        OR INT(RLIKE(LOWER(payload), 'bingbot') )>0
        OR INT(RLIKE(LOWER(payload), 'md5') )>0
        OR INT(RLIKE(LOWER(payload), 'jpg(.*?)http(.*?)1.1') )>0
        OR INT(RLIKE(LOWER(payload), 'count(.*?)cgi(.*?)http') )>0
        OR INT(RLIKE(LOWER(payload), 'this(.*?)program(.*?)can') )>0
        OR INT(RLIKE(LOWER(payload), 'sleep(.*?)sleep') )>0
        OR INT(RLIKE(LOWER(payload), 'and(.*?)sleep') )>0
        OR INT(RLIKE(LOWER(payload), 'delete'))>0
        OR INT(RLIKE(LOWER(payload), 'get(.*?)ping') )>0
        OR INT(RLIKE(LOWER(payload), 'msadc(.*?)dll(.*?)http') )>0
        OR INT(RLIKE(LOWER(payload), 'filename(.*?)asp') )>0
        OR INT(RLIKE(LOWER(payload), 'filename(.*?)jsp') )>0
                ,1, 0) AS ips_00001_payload_word_comb_03,

        IF(INT(RLIKE(LOWER(payload), 'wp(.*?)login') )>0
        OR INT(RLIKE(LOWER(payload), 'wp(.*?)content') )>0
        OR INT(RLIKE(LOWER(payload), 'wp(.*?)include') )>0
        OR INT(RLIKE(LOWER(payload), 'wp(.*?)config') )>0
                ,1, 0) AS ips_00001_payload_wp_comb_01,

        IF(INT(RLIKE(LOWER(payload), 'cmd(.*?)open') )>0
        OR INT(RLIKE(LOWER(payload), 'echo(.*?)shellshock') )>0
        OR INT(RLIKE(LOWER(payload), 'php(.*?)echo') )>0
        OR INT(RLIKE(LOWER(payload), 'admin(.*?)php') )>0
        OR INT(RLIKE(LOWER(payload), 'script(.*?)setup(.*?)php') )>0
        OR INT(RLIKE(LOWER(payload), 'phpinfo') )>0
        OR INT(RLIKE(LOWER(payload), 'administrator') )>0
        OR INT(RLIKE(LOWER(payload), 'phpmyadmin') )>0
        OR INT(RLIKE(LOWER(payload), 'access') )>0
        OR INT(RLIKE(LOWER(payload), 'eval') )>0
        OR INT(RLIKE(LOWER(payload), 'mdb') )>0
        OR INT(RLIKE(LOWER(payload), 'wise(.*?)survey(.*?)admin') )>0
        OR INT(RLIKE(LOWER(payload), 'admin(.*?)serv(.*?)admpw') )>0
        OR INT(RLIKE(LOWER(payload), 'php(.*?)create(.*?)function') )>0
                ,1, 0) AS ips_00001_payload_word_comb_04,

        IF(INT(RLIKE(LOWER(payload), 'user-agent(.*?)zgrab') )>0
        OR INT(RLIKE(LOWER(payload), 'user-agent(.*?)nmap') )>0
        OR INT(RLIKE(LOWER(payload), 'user-agent(.*?)dirbuster') )>0
        OR INT(RLIKE(LOWER(payload), 'user-agent(.*?)ahrefsbot') )>0
        OR INT(RLIKE(LOWER(payload), 'user-agent(.*?)baiduspider') )>0
        OR INT(RLIKE(LOWER(payload), 'user-agent(.*?)mj12bot') )>0
        OR INT(RLIKE(LOWER(payload), 'user-agent(.*?)petalbot') )>0
        OR INT(RLIKE(LOWER(payload), 'user-agent(.*?)semrushbot') )>0
        OR INT(RLIKE(LOWER(payload), 'user-agent(.*?)curl/') )>0
        OR INT(RLIKE(LOWER(payload), 'user-agent(.*?)masscan') )>0
        OR INT(RLIKE(LOWER(payload), 'user-agent(.*?)sqlmap') )>0
        OR INT(RLIKE(LOWER(payload), 'user-agent(.*?)urlgrabber(.*?)yum') )>0
                ,1, 0) AS ips_00001_payload_useragent_comb,
                
        (SIZE(SPLIT(LOWER(payload), 'get(.*?)http/1.')) -1)
            + (SIZE(SPLIT(LOWER(payload), 'post(.*?)http/1.')) -1)
        + (SIZE(SPLIT(LOWER(payload), 'head(.*?)http/1.')) -1)
        + (SIZE(SPLIT(LOWER(payload), 'option(.*?)http/1.')) -1)
        AS ips_00001_payload_whitelist
    FROM table
    
"""

# new_sql_query의 ips_00001_payload_base64 부터 ips_00001_payload_useragent_comb 까지 추출
# re.S의 경우, 줄바꿈 문자열 까지 매치 !!!!!!!
attack_new_sql_query = re.findall(r'ips_00001_payload_base64.*?ips_00001_payload_useragent_comb', new_sql_query, re.S)[0]


# new_sql_query의 '' 안에 있는 문자열들을 추출하여 리스트 생성, 
ai_field = re.findall(r'\'(.*?)\'', attack_new_sql_query)

           
# attack_new_sql_query 에서 'AS' 를 기준으로 분할
attack_new_sql_query_split = attack_new_sql_query.split('AS')

sql_1 = attack_new_sql_query_split[0]
sql_2 = attack_new_sql_query_split[1]
sql_3 = attack_new_sql_query_split[2]
xss = attack_new_sql_query_split[3]
cmd = attack_new_sql_query_split[4]
log4j = attack_new_sql_query_split[5]
word_1 = attack_new_sql_query_split[6]
word_2 = attack_new_sql_query_split[7]
word_3 = attack_new_sql_query_split[8]
wp = attack_new_sql_query_split[9]
word_4 = attack_new_sql_query_split[10]
user_agent = attack_new_sql_query_split[11]

sql_1 = re.findall(r'\'(.*?)\'', sql_1)
sql_2 = re.findall(r'\'(.*?)\'', sql_2)
sql_3 = re.findall(r'\'(.*?)\'', sql_3)
xss = re.findall(r'\'(.*?)\'', xss)
cmd = re.findall(r'\'(.*?)\'', cmd)
log4j = re.findall(r'\'(.*?)\'', log4j)
word_1 = re.findall(r'\'(.*?)\'', word_1)
word_2 = re.findall(r'\'(.*?)\'', word_2)
word_3 = re.findall(r'\'(.*?)\'', word_3)
wp = re.findall(r'\'(.*?)\'', wp)
word_4 = re.findall(r'\'(.*?)\'', word_4)
user_agent = re.findall(r'\'(.*?)\'', user_agent)

# device = torch.device('mps')
# device = torch.device('cpu')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

IPS_pytorch_bert_model = torch.load(IPS_pytorch_bert_model_path, map_location = device)
# IPS_pytorch_bert_explainer = torch.load(IPS_pytorch_bert_explainer_path)

model_checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

print('BERT 전이학습 모델 평가: ', IPS_pytorch_bert_model.eval())



# define a prediction function
def bert_predict(x):
    tv = torch.tensor([tokenizer.encode(v, padding='max_length', max_length=100, truncation=True) for v in x]).to(device)

    #outputs = model(tv)[0].detach().cpu().numpy()
    outputs = IPS_pytorch_bert_model(tv)[0].detach().cpu().numpy()

    scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
    val = sp.special.logit(scores[:,1]) # use one vs rest logit units
    # val = shap_logit(val)
    
    return val

bert_pipe = pipeline(task = "text-classification",
                model = IPS_pytorch_bert_model,
                tokenizer = tokenizer
                )

bert_pipe.device = device
bert_pipe.model.to(device)





# PostgreSQL - Payload 예측 DB 연동
conn = pg2.connect('''host = ?? 
                    dbname = ?? 
                    user = ?? 
                    password = ?? 
                    port = ??''')

