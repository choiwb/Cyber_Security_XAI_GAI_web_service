

import re
import numpy as np
from DSS_IPS_preprocess import *
from setting import *

from flask import Flask, render_template, request
import shap

import datetime
import pandas.io.sql as psql

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ssdeep
import tlsh
from fuzzywuzzy import fuzz
from lime.lime_text import LimeTextExplainer
import time



app = Flask(__name__)
@app.route('/')
def user_input():
    return render_template('user_input.html')


# @app.route('/web_UI_preprocess', methods = ['GET'])
def web_UI_preprocess():
    
    payload_df = predict_UI_sql_result()
    payload_arr = np.array(payload_df)

    return payload_arr, payload_df


@app.route('/web_UI_predict', methods=['POST'])
def web_UI_predict():
    
    # payload 입력
    payload_input = request.form['raw_data_str']

    # payload hash 변환 (ssdeep 이용)
    payload_hash = ssdeep.hash(payload_input)

    # payload 입력 시간
    kor_time = datetime.datetime.now()
    db_event_time = kor_time.strftime("%Y%m%d%H%M")

    sql_result_total = web_UI_preprocess() 

    payload_df = sql_result_total[1]
    payload_arr = np.array(payload_df)

    pred = IPS_model.predict(payload_arr)
    
    if pred == 1:
        db_ai = 'Attack'
    else:
        db_ai = 'Normal'

    pred_proba = IPS_model.predict_proba(payload_arr)
    
    Normal_proba = int(np.round(pred_proba[:, 0], 2) * 100)
    Attack_proba = int(np.round(pred_proba[:, 1], 2) * 100)

    db_proba = float(np.round(pred_proba[:, 1], 2))

 
    cur = conn.cursor()

    # db_total_cols = payload_df.shape[1] + 5 # 5: db_dvent_time, payload_input, db_ai, db_proba, user_opinion
    # db_total_cols = 7 # event_time, payload_input, feature_names, feature_values, ai, proba, user_opinion
    db_total_cols = 8 # event_time, payload_input, payload_hash, feature_names, feature_values, ai, proba, user_opinion

    db_input_str = ','.join(['%s'] * db_total_cols)

    # insert_query = '''insert into ips.payload_predict
    #               values ({});
    #               '''.format(db_input_str)

    insert_query = '''insert into ips.payload_predict_2
                   values ({});
                   '''.format(db_input_str)

    
    ''''''
    # payload_arr = payload_arr.T # payload_arr Transpose 
    ''''''

    event_time_list = [db_event_time]
    payload_raw_list = [payload_input]
    payload_hash_list = [payload_hash]

    # db_payload_arr = [int(i) for i in payload_arr]
    feature_list = ['|'.join(payload_df.columns.tolist())]
    feature_value_list = ['|'.join(i) for i in payload_arr.astype(str)]

    ai_list = [db_ai]
    proba_list = [db_proba]
    
    # opinion_list = ['사용자 의견 작성 필요'] # 사용자 의견 컬럼의 경우 웹 UI 상 기능 연동 필요 !!!
    user_opinion = '보안 도메인 분석가 의견란 작성 필요함'
    opinion_list = [user_opinion]

    # insert_record = event_time_list + payload_raw_list + db_payload_arr + ai_list + proba_list + opinion_list
    # insert_record = event_time_list + payload_raw_list + feature_list + feature_value_list + ai_list + proba_list + opinion_list
    insert_record = event_time_list + payload_raw_list + payload_hash_list + feature_list + feature_value_list + ai_list + proba_list + opinion_list

    print(len(insert_record))

    cur.execute(insert_query, insert_record)
    conn.commit()

    
    
    '''ssdeep 기반 payload 유사도 측정 logic'''
    select_query = '''select * from ips.payload_predict'''

    payload_predict_db = psql.read_sql(select_query, conn)

    # 입력된 payload row 제외 필요!
    similar_df = payload_predict_db[['payload_input', 'payload_hash', 'ai', 'proba']].iloc[0:-1, :]
    # fuzz.ratio
    similar_df['fuzz_total'] = 0
    # fuzz.partial_ratio
    similar_df['fuzz_part'] = 0
    # ssdeep.compare
    similar_df['compare'] = 0

    # payload DB hash 값 들과 입력된 payload hash 값 비교 및 유사도 측정
    similar_df['fuzz_total'] = similar_df.apply(lambda x: fuzz.ratio(payload_hash, x['payload_hash']), axis = 1)
    similar_df['fuzz_part'] = similar_df.apply(lambda x: fuzz.partial_ratio(payload_hash, x['payload_hash']), axis = 1)
    similar_df['compare'] = similar_df.apply(lambda x: ssdeep.compare(payload_hash, x['payload_hash']), axis = 1)

    # 입력된 payload의 최대 fuzz total 유사도
    max_fuzz_total = max(similar_df['fuzz_total'])
    print("입력된 payload의 최대 fuzz total 유사도 : ", max_fuzz_total)

    # 입력된 payload의 최대 fuzz total 유사도 payload db에서 선택
    max_fuzz_total_payload_df = similar_df[similar_df['fuzz_total'] == max_fuzz_total]
    max_fuzz_total_payload_df = max_fuzz_total_payload_df.drop_duplicates(subset = ['payload_input', 'payload_hash', 'fuzz_total'])
    max_fuzz_total_payload = max_fuzz_total_payload_df.iloc[0,0]
    print("입력된 payload의 최대 fuzz total 유사도 payload : ", max_fuzz_total_payload)
    max_fuzz_total_payload_ai = max_fuzz_total_payload_df.iloc[0,2]
    print("입력된 payload의 최대 fuzz total 유사도 payload 예측 라벨 : ", max_fuzz_total_payload_ai)
    max_fuzz_total_payload_proba = max_fuzz_total_payload_df.iloc[0,3]
    print("입력된 payload의 최대 fuzz total 유사도 payload 예측 확률 : ", max_fuzz_total_payload_proba)


    # 입력된 payload의 최대 fuzz part 유사도
    max_fuzz_part = max(similar_df['fuzz_part'])
    print("입력된 payload의 최대 fuzz part 유사도 : ", max_fuzz_part)

    # 입력된 payload의 최대 fuzz total 유사도 payload db에서 선택
    max_fuzz_part_payload_df = similar_df[similar_df['fuzz_part'] == max_fuzz_part]
    max_fuzz_part_payload_df = max_fuzz_part_payload_df.drop_duplicates(subset = ['payload_input', 'payload_hash', 'fuzz_part'])
    max_fuzz_part_payload = max_fuzz_part_payload_df.iloc[0,0]
    print("입력된 payload의 최대 fuzz part 유사도 payload : ", max_fuzz_part_payload)
    max_fuzz_part_payload_ai = max_fuzz_total_payload_df.iloc[0,2]
    print("입력된 payload의 최대 fuzz part 유사도 payload 예측 라벨 : ", max_fuzz_part_payload_ai)
    max_fuzz_part_payload_proba = max_fuzz_total_payload_df.iloc[0,3]
    print("입력된 payload의 최대 fuzz part 유사도 payload 예측 확률 : ", max_fuzz_part_payload_proba)


    # 입력된 payload의 최대 compare 유사도
    max_compare = max(similar_df['compare'])
    print("입력된 payload의 최대 compare 유사도 : ", max_compare)         

    # 입력된 payload의 최대 compare 유사도 payload db에서 선택
    max_compare_payload_df = similar_df[similar_df['compare'] == max_compare]
    max_compare_payload_df = max_compare_payload_df.drop_duplicates(subset = ['payload_input', 'payload_hash', 'compare'])
    max_compare_payload = max_compare_payload_df.iloc[0,0]
    print("입력된 payload의 최대 compare 유사도 payload : ", max_compare_payload)
    max_compare_payload_ai = max_compare_payload_df.iloc[0,2]
    print("입력된 payload의 최대 compare 유사도 payload 예측 라벨 : ", max_compare_payload_ai)
    max_compare_payload_proba = max_compare_payload_df.iloc[0,3]
    print("입력된 payload의 최대 compare 유사도 payload 예측 확률 : ", max_compare_payload_proba)


    # 입력된 payload의 유사도 측정 검증 (fuzz_ratio, fuzz_partial_ratio, ssdeep_compare)
    fuzz_ratio = fuzz.ratio(payload_hash, payload_hash)
    fuzz_part_ratio = fuzz.partial_ratio(payload_hash, payload_hash)
    ssdeep_compare = ssdeep.compare(payload_hash, payload_hash)

    if fuzz_ratio == 100 & fuzz_part_ratio == 100 & ssdeep_compare == 100:
        print('입력된 payload의 자기 유사성이 3가지 유사도 측정 방법 모두 100 임.')
    else:
        print('입력된 payload의 자기 유사성이 3가지 유사도 측정 방법 따라 다름 !!!!!!!')
     

    return render_template('server_output.html', data = [pred, Normal_proba, Attack_proba])


############################################
# logit (log odds) 형태를 확률로 변환
def shap_logit(x):
    logit_result = 1 / (1 + np.exp(-x))
    return logit_result

# 텍스트 전처리
def text_prep(x):
    x = x.lower()
    x = re.sub(r'[^a-z]+', ' ', x)
    return x


###############################################

signature_list = ['/etc/passwd', 'password=admin']
# 탐지 패턴 소문자화
signature_list = [x.lower() for x in signature_list]

method_list = ['IGLOO-UD-File Downloading Vulnerability-1(/etc/passwd)', 'IGLOO-UD-WeakIDPasswd-1(password=admin)']

# AI 생성 필드 리스트 (domain 기반 표준피처) - 단, 화이트리스트 피처 및 base64 관련 피처는 제외
ai_field = ['select(.*?)from', 'select(.*?)count', 'select(.*?)distinct', 'union(.*?)select', 'select(.*?)extractvalue(.*?)xmltype',
           'from(.*?)generate(.*?)series', 'from(.*?)group(.*?)by', 'case(.*?)when', 'then(.*?)else', 'waitfor(.*?)delay', 'db(.*?)sql(.*?)server',
           'cast(.*?)chr', 'like', 'upper(.*?)xmltype', 'script(.*?)alert', 'wget(.*?)ttp', 'chmod(.*?)777', 'rm(.*?)rf', 'cd(.*?)tmp',
           'jndi(.*?)dap', 'jndi(.*?)dns', 'etc(.*?)passwd', 'document(.*?)createelement', 'cgi(.*?)bin', 'document(.*?)forms', 'document(.*?)location',
           'fckeditor(.*?)filemanager', 'manager(.*?)html', 'current_config(.*?)passwd', 'currentsetting(.*?)htm', 'well(.*?)known',
           'bash(.*?)history', 'apache(.*?)struts', 'document(.*?)open', 'backup(.*?)sql', 'robots(.*?)txt', 'sqlexec(.*?)php',
           'exec', 'htaccess', 'htpasswd', 'cgi(.*?)cgi', 'api(.*?)ping', 'aaaaaaaaaa', 'cacacacaca', 'mozi',
           'bingbot', 'md5', 'jpg(.*?)http(.*?)1.1', 'count(.*?)cgi(.*?)http', 'this(.*?)program(.*?)can', 'sleep(.*?)sleep', 'and(.*?)sleep',
           'delete', 'get(.*?)ping', 'msadc(.*?)dll(.*?)http', 'filename(.*?)asp', 'filename(.*?)jsp',
           'wp(.*?)login', 'wp(.*?)content', 'wp(.*?)include', 'wp(.*?)config', 'cmd(.*?)open', 'echo(.*?)shellshock', 'php(.*?)echo',
           'echo', 'admin(.*?)php', 'script(.*?)setup(.*?)php', 'phpinfo', 'adminostrator', 'phpmyadmin', 'access', 'passwd', 'eval', 'php', 'cmd', 'mdb',
           'wise(.*?)survey(.*?)admin', 'admin(.*?)serv(.*?)admpw', 'php(.*?)create(.*?)function',
           'user-agent(.*?)zgrab', 'user-agent(.*?)nmap', 'user-agent(.*?)dirbuster', 'user-agent(.*?)ahrefsbot',
           'user-agent(.*?)baiduspider', 'user-agent(.*?)mj12bot', 'user-agent(.*?)petalbot',
           'user-agent(.*?)semrushbot', 'user-agent(.*?)curl', 'user-agent(.*?)masscan', 'user-agent(.*?)sqlmap',
           'user-agent(.*?)urlgrabber(.*?)yum']

# IPS & WAF 피처 설명 테이블 생성
ips_feature_df = pd.DataFrame([['ips_00001_payload_base64', 'payload에 공격관련 키워드(base64)가 포함되는 경우에 대한 표현'],
                                ['ips_00001_payload_cmd_comb_01', 'payload에 cmd 관련 키워드 조합이 포함되는 경우에 대한 표현'],
                                ['ips_00001_payload_log4j_comb_01', 'payload에 log4j 관련 키워드 조합이 포함되는 경우에 대한 표현'],
                                ['ips_00001_payload_sql_comb_01', 'payload에 SQL-I 관련 키워드 조합이 포함되는 경우에 대한 표현'],
                                ['ips_00001_payload_sql_comb_02', 'payload에 SQL-I 관련 키워드 조합이 포함되는 경우에 대한 표현'],
                                ['ips_00001_payload_sql_comb_03', 'payload에 SQL-I 관련 키워드 또는 키워드 조합이 포함되는 경우에 대한 표현'],
                                ['ips_00001_payload_useragent_comb', 'payload에 악성 user_agent가 포함되는 경우에 대한 표현'],
                                ['ips_00001_payload_word_comb_01', 'payload에 공격관련 키워드 조합이 포함되는 경우에 대한 표현'],
                                ['ips_00001_payload_word_comb_02', 'payload에 공격관련 키워드 또는 키워드 조합이 포함되는 경우에 대한 표현'],
                                ['ips_00001_payload_word_comb_03', 'payload에 공격관련 키워드 또는 키워드 조합이 포함되는 경우에 대한 표현'],
                                ['ips_00001_payload_word_comb_04', 'payload에 공격관련 키워드 또는 키워드 조합이 포함되는 경우에 대한 표현'],
                                ['ips_00001_payload_wp_comb_01', 'payload에 wp 관련 키워드 조합이 포함되는 경우에 대한 표현'],
                                ['ips_00001_payload_xss_comb_01', 'payload에 XSS 관련 키워드 조합이 포함되는 경우에 대한 표현'],
                                ['ips_00001_payload_whitelist', 'payload에 공격과 관련없이 로그전송 이벤트인 경우에 대한 표현']
                              ]
                                , columns=['피처 명', '피처 설명'])


def highlight_text(text, signature, ai_field):
    # background yellow - 시그니처 패턴
    replacement = "\033[103m" + "\\1" + "\033[49m"
    # foreground red - AI 생성 필드
    replacement_2 = "\033[91m" + "\\1" + "\033[39m"

    # 시그니처 패턴 또는 AI 생성 필드 인 경우, highlight 처리
    # re.escape() : 특수문자를 이스케이프 처리
    text = re.sub("(" + "|".join(map(re.escape, signature)) + ")", replacement, text, flags=re.I)
    text = re.sub("(" + "|".join(ai_field) + ")", replacement_2, text, flags=re.I)

    regex = re.compile('\x1b\[103m(.*?)\x1b\[49m')

    matches = [regex.match(text[i:]) for i in range(len(text))] 
    sig_pattern_prep = [m.group(0) for m in matches if m] 

    sig_pattern = [re.sub(r'\x1b\[103m|\x1b\[49m', '', i) for i in sig_pattern_prep]
    sig_pattern = [re.sub(r'\x1b\[91m|\x1b\[39m', '', i) for i in sig_pattern]

    sig_pattern_df = pd.DataFrame(columns = ['탐지 순서', '탐지 명'])
    count = 0
    for i in sig_pattern:
        # 탐지 패턴 소문자화
        i = i.lower()
        count = count + 1

        if i in signature_list:
            j = signature_list.index(i)
            # print('%d 번째 시그니처 패턴 공격명: %s' %(count, method_list[j]))
            one_row_df = pd.DataFrame([[count, method_list[j]]], columns = ['탐지 순서', '탐지 명'])
            sig_pattern_df = pd.concat([sig_pattern_df, one_row_df], axis = 0)

    return text, sig_pattern_df

@app.route('/XAI_result', methods = ['POST'])
def XAI_result():

   # payload의 raw data 입력 값!
    raw_data_str = request.form['raw_data_str']

    # XAI 실행 시간
    kor_time = datetime.datetime.now()
    xai_event_time = kor_time.strftime("%Y%m%d%H%M")
    
    sql_result_total = web_UI_preprocess()

    payload_df = sql_result_total[1]
    payload_arr = np.array(payload_df)
    
    IPS_total_explainer = pickle.load(open(IPS_total_explainer_path, 'rb'))
    expected_value_sql = IPS_total_explainer.expected_value
    expected_value_sql = np.array(expected_value_sql)
    expected_value_sql_logit = shap_logit(expected_value_sql)
    print('sql SHAP 기댓값 (logit 적용 함): ', expected_value_sql_logit)

    # anomalies : shap_values[1], normal: shap_values[0]
    shap_values_sql = IPS_total_explainer.shap_values(payload_arr)
    shap_values_sql = np.array(shap_values_sql)
    shap_values_sql_logit = shap_logit(shap_values_sql)
    print('sql SHAP values (logit 적용 함): ', shap_values_sql_logit)

    # mean_shap_values = np.abs(shap_values).mean(0)
    # mean_shap_values = np.abs(shap_values[1]).mean(0)
    mean_shap_values = np.abs(shap_values_logit).mean(0)
    # 예측 데이터는 1건이므로, 반드시 평균을 구할 필요가 없음 !!!!!

    mean_shap_value_df = pd.DataFrame(list(zip(payload_df.columns, mean_shap_values)),
                                   columns=['피처 명','피처 중요도'])

    pred = IPS_model.predict(payload_arr)
    if pred == 1:
        db_ai = 'anomalies'
    else:
        db_ai = 'normal'
    
    if db_ai == 'anomalies':
        mean_shap_value_df.sort_values(by=['피처 중요도'],
                                    ascending=False, inplace=True)
    else:
        mean_shap_value_df.sort_values(by=['피처 중요도'],
                                    ascending=True, inplace=True)
    
    top10_shap_values = mean_shap_value_df.iloc[0:10, :]
    top10_shap_values = top10_shap_values.reset_index(drop = True)

    top10_shap_values.index = top10_shap_values.index + 1
    top10_shap_values = top10_shap_values.reset_index(drop = False)
    top10_shap_values = top10_shap_values.rename(columns = {'index': '순위'})
    # print(top10_shap_values)

    # 피처 설명 테이블과 join
    top10_shap_values = pd.merge(top10_shap_values, ips_feature_df, how = 'left', on = '피처 명')
    top10_shap_values = top10_shap_values[['순위', '피처 명', '피처 설명', '피처 중요도']]

    payload_df_t = payload_df.T
    payload_df_t.columns = ['피처 값']
    # payload_df_t에 피처 명 컬럼 추가
    payload_df_t['피처 명'] = payload_df_t.index
    top10_shap_values = pd.merge(top10_shap_values, payload_df_t, how = 'left', on = '피처 명')
    top10_shap_values = top10_shap_values[['순위', '피처 명', '피처 설명', '피처 값', '피처 중요도']]

    # 소수점 4째 자리까지 표현
    top10_shap_values['피처 중요도'] = top10_shap_values['피처 중요도'].apply(lambda x: round(x, 4))
    top10_shap_values['피처 설명'] = top10_shap_values['피처 설명'].fillna('payload에서 TF-IDF 기반 추출된 키워드에 대한 표현')

    
    ##################################################
    # 학습 데이터 기반 피처 중요도 요약 (상위 3개 피처)
    ##################################################
    
    # TF-IDF 피처에 대한 설명 필요 !!!!!!!!!!!!!!!!
    
    first_feature = top10_shap_values.iloc[0, 1]
    first_fv = top10_shap_values.iloc[0, 3]
    second_feature = top10_shap_values.iloc[1, 1]
    second_fv = top10_shap_values.iloc[1, 3]
    third_feature = top10_shap_values.iloc[2, 1]
    third_fv = top10_shap_values.iloc[2, 3]


    if first_feature.startswith('ips_'):
        if first_feature != 'ips_00001_payload_whitelist':
            if first_fv == 1:
                first_fv_result = '공격 탐지'
            else:
                first_fv_result = '정상 인식'

            first_fv_df = ips_training_data[ips_training_data[first_feature] == first_fv]
            first_fv_df_anomalies = first_fv_df[first_fv_df['label'] == 1]
            first_fv_df_anomalies_ratio = first_fv_df_anomalies.shape[0] / first_fv_df.shape[0]
            first_fv_df_anomalies_ratio = round(first_fv_df_anomalies_ratio * 100, 2)
            first_fv_df_normal_ratio = 100 - first_fv_df_anomalies_ratio

            first_statement = '%s 가 %s 하였고, 헉숩 데이터에서 해당 피처 값은 정탐: %.2f%% 오탐: %.2f%% 입니다.' %(first_feature, first_fv_result, first_fv_df_anomalies_ratio, first_fv_df_normal_ratio)
        else:
            first_statement = '로그 전송 이벤트가 %d건 입니다.' % first_fv
    else:
        if first_fv >  0:
            first_word = first_feature[8:]
            first_fv_df = ips_training_data[ips_training_data[first_feature] > 0]
            first_fv_df_anomalies = first_fv_df[first_fv_df['label'] == 1]
            first_fv_df_anomalies_ratio = first_fv_df_anomalies.shape[0] / first_fv_df.shape[0]
            first_fv_df_anomalies_ratio = round(first_fv_df_anomalies_ratio * 100, 2)
            first_fv_df_normal_ratio = 100 - first_fv_df_anomalies_ratio

            first_statement = '%s 키워드가 1번 이상 등장하였고, 헉숩 데이터에서 해당 키워드가 1번 이상 등장한 경우, 정탐: %.2f%% 오탐: %.2f%% 입니다.' %(first_word, first_fv_df_anomalies_ratio, first_fv_df_normal_ratio)
        else:
            first_word = first_feature[8:]
            first_fv_df = ips_training_data[ips_training_data[first_feature] == 0]
            first_fv_df_anomalies = first_fv_df[first_fv_df['label'] == 1]
            first_fv_df_anomalies_ratio = first_fv_df_anomalies.shape[0] / first_fv_df.shape[0]
            first_fv_df_anomalies_ratio = round(first_fv_df_anomalies_ratio * 100, 2)
            first_fv_df_normal_ratio = 100 - first_fv_df_anomalies_ratio
            
            first_statement = '%s 키워드가 등장하지 않았고, 헉숩 데이터에서 해당 키워드가 등장하지 않은 경우, 정탐: %.2f%% 오탐: %.2f%% 입니다.' %(first_word, first_fv_df_anomalies_ratio, first_fv_df_normal_ratio)


    if second_feature.startswith('ips_'):
        if second_feature != 'ips_00001_payload_whitelist':
            if second_fv == 1:
                second_fv_result = '공격 탐지'
            else:
                second_fv_result = '정상 인식'

            second_fv_df = ips_training_data[ips_training_data[second_feature] == second_fv]
            second_fv_df_anomalies = second_fv_df[second_fv_df['label'] == 1]
            second_fv_df_anomalies_ratio = second_fv_df_anomalies.shape[0] / second_fv_df.shape[0]
            second_fv_df_anomalies_ratio = round(second_fv_df_anomalies_ratio * 100, 2)
            second_fv_df_normal_ratio = 100 - second_fv_df_anomalies_ratio

            second_statement = '%s 가 %s 하였고, 헉숩 데이터에서 해당 피처 값은 정탐: %.2f%% 오탐: %.2f%% 입니다.' %(second_feature, second_fv_result, second_fv_df_anomalies_ratio, second_fv_df_normal_ratio)
        else:
            second_statement = '로그 전송 이벤트가 %d건 입니다.' % second_fv
    else:
        if second_fv > 0:
            second_word = second_feature[8:]
            second_fv_df = ips_training_data[ips_training_data[second_feature] > 0]
            second_fv_df_anomalies = second_fv_df[second_fv_df['label'] == 1]
            second_fv_df_anomalies_ratio = second_fv_df_anomalies.shape[0] / second_fv_df.shape[0]
            second_fv_df_anomalies_ratio = round(second_fv_df_anomalies_ratio * 100, 2)
            second_fv_df_normal_ratio = 100 - second_fv_df_anomalies_ratio

            second_statement = '%s 키워드가 1번 이상 등장하였고, 헉숩 데이터에서 해당 키워드가 1번 이상 등장한 경우, 정탐: %.2f%% 오탐: %.2f%% 입니다.' %(second_word, second_fv_df_anomalies_ratio, second_fv_df_normal_ratio)
        else:
            second_word = second_feature[8:]
            second_fv_df = ips_training_data[ips_training_data[second_feature] == 0]
            second_fv_df_anomalies = second_fv_df[second_fv_df['label'] == 1]
            second_fv_df_anomalies_ratio = second_fv_df_anomalies.shape[0] / second_fv_df.shape[0]
            second_fv_df_anomalies_ratio = round(second_fv_df_anomalies_ratio * 100, 2)
            second_fv_df_normal_ratio = 100 - second_fv_df_anomalies_ratio
            
            second_statement = '%s 키워드가 등장하지 않았고, 헉숩 데이터에서 해당 키워드가 등장하지 않은 경우, 정탐: %.2f%% 오탐: %.2f%% 입니다.' %(second_word, second_fv_df_anomalies_ratio, second_fv_df_normal_ratio)


    if third_feature.startswith('ips_'):
        if third_feature != 'ips_00001_payload_whitelist':
            if third_fv == 1:
                third_fv_result = '공격 탐지'
            else:
                third_fv_result = '정상 인식'
            third_fv_df = ips_training_data[ips_training_data[third_feature] == third_fv]
            third_fv_df_anomalies = third_fv_df[third_fv_df['label'] == 1]
            third_fv_df_anomalies_ratio = third_fv_df_anomalies.shape[0] / third_fv_df.shape[0]
            third_fv_df_anomalies_ratio = round(third_fv_df_anomalies_ratio * 100, 2)
            third_fv_df_normal_ratio = 100 - third_fv_df_anomalies_ratio

            third_statement = '%s 가 %s 하였고, 헉숩 데이터에서 해당 피처 값은 정탐: %.2f%% 오탐: %.2f%% 입니다.' %(third_feature, third_fv_result, third_fv_df_anomalies_ratio, third_fv_df_normal_ratio)
        else:
            third_statement = '로그 전송 이벤트가 %d건 입니다.' % third_fv
    else:
        if third_fv > 0:
            third_word = third_feature[8:]
            third_fv_df = ips_training_data[ips_training_data[third_feature] > 0]
            third_fv_df_anomalies = third_fv_df[third_fv_df['label'] == 1]
            third_fv_df_anomalies_ratio = third_fv_df_anomalies.shape[0] / third_fv_df.shape[0]
            third_fv_df_anomalies_ratio = round(third_fv_df_anomalies_ratio * 100, 2)
            third_fv_df_normal_ratio = 100 - third_fv_df_anomalies_ratio

            third_statement = '%s 키워드가 1번 이상 등장하였고, 헉숩 데이터에서 해당 키워드가 1번 이상 등장한 경우, 정탐: %.2f%% 오탐: %.2f%% 입니다.' %(third_word, third_fv_df_anomalies_ratio, third_fv_df_normal_ratio)
        else:
            third_word = third_feature[8:]
            third_fv_df = ips_training_data[ips_training_data[third_feature] == 0]
            third_fv_df_anomalies = third_fv_df[third_fv_df['label'] == 1]
            third_fv_df_anomalies_ratio = third_fv_df_anomalies.shape[0] / third_fv_df.shape[0]
            third_fv_df_anomalies_ratio = round(third_fv_df_anomalies_ratio * 100, 2)
            third_fv_df_normal_ratio = 100 - third_fv_df_anomalies_ratio

            third_statement = '%s 키워드가 등장하지 않았고, 헉숩 데이터에서 해당 키워드가 등장하지 않은 경우, 정탐: %.2f%% 오탐: %.2f%% 입니다.' %(third_word, third_fv_df_anomalies_ratio, third_fv_df_normal_ratio)

    
    
    
    # top10_shap_values to html
    top10_shap_values_html = top10_shap_values.to_html(index=False, justify='center')

    force_plot = shap.force_plot(expected_value_sql, shap_values_sql, payload_arr, link = 'logit',
                        feature_names = payload_df.columns,
                        matplotlib = False)
            
    force_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"

    xai_shap_save_path = 'SHAP force plot 결과 저장 경로'
    # SHAP's force plot의 html 저장
    # shap.save_html(os.path.join(xai_shap_save_path, 'shap_force_plot_%s.html' %(xai_event_time)), force_plot)

    # SHAP's force plot의 이미지 (png) 저장 => link = 'logit' 의 경우 파라미터 적용 안됨 !!!!!
    # shap.force_plot(expected_value, shap_values[1], payload_df,
    #                   matplotlib = True, show = False)
    # plt.savefig(os.path.join(xai_shap_save_path, 'shap_force_plot_%s.png' %(xai_event_time)),
    #                        bbox_inches = 'tight', dpi = 700)


    #############################################    
    # SHAP's force plot - text feature

    payload_str_df = pd.DataFrame([raw_data_str], columns = ['payload'])
    payload_str = payload_str_df['payload']

    payload_test_tfidf = IPS_text_model['tfidfvectorizer'].transform(payload_str).toarray()
    
    IPS_text_explainer = shap.TreeExplainer(IPS_text_model['lgbmclassifier'],
                   feature_names=IPS_text_model['tfidfvectorizer'].get_feature_names_out())
    
    
    expected_value_text = IPS_text_explainer.expected_value
    expected_value_text = np.array(expected_value_text)
    expected_value_text_logit = shap_logit(expected_value_text)
    print('text SHAP 기댓값 (logit 적용 함): ', expected_value_text_logit)

    shap_values_text = IPS_text_explainer.shap_values(payload_test_tfidf)
    shap_values_text = np.array(shap_values_text)
    shap_values_text_logit = shap_logit(shap_values_text)
    print('text SHAP values (logit 적용 함): ', shap_values_text_logit)

    text_plot = shap.force_plot(expected_value_text, shap_values_text[1], link = 'logit',
                                feature_names = IPS_text_model['tfidfvectorizer'].get_feature_names_out(),
                                matplotlib = False)
    text_explainer_html = f"<head>{shap.getjs()}</head><body>{text_plot.html()}</body>"


    #############################################    
    # LIME TextTabularExplainer
    # 0: normal, 1: anomalies
    class_names = ['normal', 'anomalies']
    
    text_explainer = LimeTextExplainer(class_names=class_names)

    pred_explainer = text_explainer.explain_instance(raw_data_str, IPS_text_model.predict_proba,
                                                num_features=10)

    pred_explainer.show_in_notebook(text=True)
    lime_text_explainer_html = pred_explainer.as_html()

    #############################################
    # PyTorch BERT explainer

    # transformers == 4.21.3 에서만 동작 (4.23.1 에서는 에러) => 20221101 기준
    # BERT 예측 라벨
    print('BERT 파이프라인 device: ', bert_pipe.device)
    # 하나의 dimension에 token 이 512개 까지만 연산 가능 512개 초과하는 경우, tensor size 에러 나므로, 알고리즘 수정 필요!
    # 입력 payload 최대 900글자 제한
    raw_data_str_short = raw_data_str[0:900]
    bert_pipe_result = bert_pipe(raw_data_str_short)

    bert_label = bert_pipe_result[0]['label']
    if bert_label == 'POSITIVE':
        bert_label = 'anomalies'
    else:
        bert_label = 'mormal'

    bert_score = bert_pipe_result[0]['score']
    bert_score = np.round(bert_score, 4)

    print('BERT 예측 라벨:' , bert_label)
    print('BERT 예측 스코어: ', bert_score)


    # raw_data_str_prep = text_prep(raw_data_str)
    # print('전처리된 payload: ', raw_data_str_prep)

    payload_test_df = pd.DataFrame([raw_data_str], columns = ['payload'])
    bert_payload = payload_test_df['payload'].sample(1)

    '''BERT 모델 호출 후 예측 속도 향상 필요!!!!!!!!!!!!!! CPU => MPS 또는 GPU'''
    bert_shap_start = time.time()
    # build an explainer using a token masker
    IPS_pytorch_bert_explainer = shap.Explainer(bert_predict, tokenizer)
    bert_shap_values = IPS_pytorch_bert_explainer(bert_payload, fixed_context=1, batch_size=1)
    bert_shap_end = time.time()
    dur_bert_shap = bert_shap_end - bert_shap_start
    # cpu 연산 시간: 11.23 초, mps 연산 시간: 7.46 초
    print('mps 연산 시간: %.2f (초)' %(dur_bert_shap))

    text_html = shap.text_plot(bert_shap_values, display = False)
    # text_html = f"<head>{shap.getjs()}</head><body>{text_plot.html()}</body>"

    # 보안 시그니처 패턴 리스트 highlight
    # sig_pattern, sig_df = highlight_text(raw_data_str, signature_list)
    sig_ai_pattern, sig_df = highlight_text(raw_data_str, signature_list, ai_field)

    print(sig_ai_pattern)
    # print(sig_df)

    # HTML 형태 payload 의 경우, 소괄호 치환 필요
    sig_ai_pattern = re.sub(r'[\\<]', r'&lt;', sig_ai_pattern)
    sig_ai_pattern = re.sub(r'[\\>]', r'&gt;', sig_ai_pattern)

    foreground_regex = r'\x1b\[91m(.*?)\x1b\[39m'
    background_regex = r'\x1b\[103m(.*?)\x1b\[49m'
    
    sig_ai_pattern = re.sub(foreground_regex, r'<font color = "red">\1</font>', sig_ai_pattern)
    sig_ai_pattern = re.sub(background_regex, r'<span style = "background-color:yellow;">\1</span>', sig_ai_pattern)
    
    sig_pattern_html = f"<head>{sig_ai_pattern}</head>"        
    sig_df_html = sig_df.to_html(index=False, justify='center')

    return render_template('XAI_output.html', payload_raw_data = request.form['raw_data_str'],  
                                force_html = force_html,
                                # waterfall_html = waterfall_html,
                                text_explainer_html = text_explainer_html,
                                lime_text_explainer_html = lime_text_explainer_html, 
                                text_html = text_html,
                                bert_label = bert_label,
                                bert_score = bert_score,
                                top10_shap_values_html = top10_shap_values_html,
                                first_statement = first_statement,
                                second_statement = second_statement,
                                third_statement = third_statement,
                                sig_pattern_html = sig_pattern_html,
                                sig_df_html = sig_df_html,
                                # summary_html = summary_html
                                )


@app.route('/WAF_payload_parsing', methods = ['POST'])
def WAF_payload_parsing():
    raw_data_str = request.form['raw_data_str']

    pre_df = pd.DataFrame([raw_data_str], columns = ['payload'])
    pre_df['http_method'] = [str(x).split(' ')[0] for x in pre_df['payload']]


    mtd = [str(x).split(' ')[0] for x in pre_df['payload']]
    for i, m in enumerate(mtd):
        if len(m) > 10 or len(m) == 1 or not m.isalpha():
            mtd[i] = ''

    method_list = ['', 'upload', 'get', 'profind', 'put', 'options', 'head', 'trace', 'connect', 'delete', 'post', 'patch']

    m_idx = []
    not_m_idx = []

    for i, m in enumerate(pre_df['http_method']):
        # if m in method_list:
        if m.lower() in method_list:
            m_idx.append(i)
        else:
            not_m_idx.append(i)

    df_m = pre_df.iloc[m_idx].reset_index(drop=True)
    df_nm = pre_df.iloc[not_m_idx].reset_index(drop=True)

    # payload_0: payload에서 ' ' (공백) 첫번째를 기준으로 나누엇, 2번째 값을 반환하므로, http_url 부터 끝 임.
    # 따라서, http_url + http_query + http_body
    df_m['payload_0'] = [str(x).split(' ', maxsplit=1)[1] for x in df_m['payload']]
    # url_query: payload_0에서, ' ' (공백) 첫번째를 기준으로 나누어, 1번째 값을 반환하므로, http_url ~ http_query 임.
    # 따라서, http_url + http_query
    df_m['url_query'] = [str(x).split(' ', maxsplit=1)[0] for x in df_m['payload_0']]

    http_body = []
    for i in df_m['payload_0']:
        if ' ' in i:
            # payload_0에서 공백이 있는 경우, http_body
            http_body.append(i.split(' ', maxsplit=1)[1])
        else:
            http_body.append('')

    df_m['http_body'] = http_body
    # url_query에서 ? 가 있는 경우, 1번째 값을 반환하므로, http_url 임.
    df_m['http_url'] = [str(x).split('?', maxsplit=1)[0] for x in df_m['url_query']]

    query = []
    for i in df_m['url_query']:
        if '?' in i:
            # url_query에서, ?가 있는 경우, 2번째 값을 반환하므로, http_query 임.
            query.append('?'+i.split('?', maxsplit=1)[1])
        else:
            query.append('')
    df_m['http_query'] = query

    

    df_res = df_m[['payload', 'http_method', 'http_url', 'http_query', 'http_body']]

    a = []
    a.append('')
    df_nm['http_method'] = a
    df_nm['http_url'] = a
    df_nm['http_query'] = a                                                         
    df_nm['http_body'] = a
    df_nm['uri'] = list(df_nm['payload'])

    if str(df_nm['http_url'][0:1]) == 'nan' and str(df_nm['http_query'][0:1]) == 'nan' and str(df_nm['http_body'][0:1]) == 'nan':
        df_nm['http_body'][0:1] = df_nm['uri'][0:1]

    if df_nm['uri'][0:1].isna().sum() == 0:
        df_nm = df_nm.fillna('-')
        df_nm_np = np.where(df_nm.iloc[:, :] == '', '-', df_nm.iloc[:, :])
        df_nm = pd.DataFrame(df_nm_np, columns = df_nm.columns.tolist())
        df_nm['http_body'] = df_nm['uri']
        df_nm = df_nm.drop(['payload', 'uri'], axis = 1)
        df_nm['http_version'] = '-'
        df_nm = df_nm[['http_method', 'http_url', 'http_query', 'http_version', 'http_body']]
        
        # http_query 필드의 첫 글자가 '?' 인 경우, '' 처리
        if df_nm.iloc[0,2].startswith('?') == True:
            df_nm['http_query'] = df_nm['http_query'].str[1:]

        # FLASK 적용
        flask_html = df_nm.to_html(index = False, justify = 'center')
        # print(flask_df)
        # CTI 적용
        cti_json = df_nm.to_json(orient = 'records')
        # print(ctf_df)
        warning_statement = '비정상적인 Payload 입력 형태 입니다. (예, payload 의 시작이 특수문자 등)'


    else:
        # http_version => HTTP/1.1 OR HTTP/1.0 OR HTTP/2.0
        df_res['http_version'] = '-'
        # df_res.iloc[0,4]) ' '  로 시작하는 경우 '' 처리
        if df_res.iloc[0,4].startswith(' ') == True:
            df_res['http_body'] = df_res['http_body'].str[1:]

        if df_res.iloc[0,4].lower().startswith('http/') == True:
            df_res['http_version'][0:1] = df_res['http_body'][0:1].str[0:8]
            df_res['http_body'] = df_res['http_body'].str[8:]

        final_df = df_res[['payload', 'http_method', 'http_url', 'http_query', 'http_version', 'http_body']]
        final_df = final_df.drop('payload', axis = 1)

        final_np = np.where(final_df.iloc[:, :] == '', '-', final_df.iloc[:, :])
        final_df = pd.DataFrame(final_np, columns = final_df.columns.tolist())

        # http_query 필드의 첫 글자가 '?' 인 경우, '' 처리
        if final_df.iloc[0,2].startswith('?') == True:
            final_df['http_query'] = final_df['http_query'].str[1:]

        # FLASK 적용
        flask_html = final_df.to_html(index = False, justify = 'center')
        # print(flask_df)
        # CTI 적용
        cti_json = final_df.to_json(orient = 'records')
        # print(ctf_df)

        warning_statement = '정상적인 Payload 입력 형태 입니다.'


    return render_template('WAF_payload_parsing_output.html',
                        # waf_raw_data_str = request.form['waf_raw_data_str'],
                        flask_html = flask_html,
                        warning_statement = warning_statement
                        )



@app.route('/WEB_payload_parsing', methods = ['POST'])
def WEB_payload_parsing():
    raw_data_str = request.form['raw_data_str']

    # raw_data_str에 '"'가 4개 이상 (2쌍) 인 경우, APACHE, 아니면, IIS
    if raw_data_str.count('"') >= 4:
    
        pre_df = pd.DataFrame([raw_data_str], columns = ['payload'])
        pre_df['payload_prep'] = [str(x).split('"', maxsplit=1)[1] for x in pre_df['payload']]

        pre_df['http_method'] = [str(x).split(' ', maxsplit=1)[0] for x in pre_df['payload_prep']]

        mtd = [str(x).split(' ', maxsplit=1)[0] for x in pre_df['payload_prep']]
        for i, m in enumerate(mtd):
            if len(m) > 10 or len(m) == 1 or not m.isalpha():
                mtd[i] = ''

        method_list = ['', 'upload', 'get', 'profind', 'put', 'options', 'head', 'trace', 'connect', 'delete', 'post', 'patch']

        m_idx = []
        not_m_idx = []

        for i, m in enumerate(pre_df['http_method']):
            # if m in method_list:
            if m.lower() in method_list:
                m_idx.append(i)
            else:
                not_m_idx.append(i)


        df_m = pre_df.iloc[m_idx].reset_index(drop=True)
        df_nm = pre_df.iloc[not_m_idx].reset_index(drop=True)

        # payload_0: payload에서 ' ' (공백) 첫번째를 기준으로 나누엇, 2번째 값을 반환하므로, http_url 부터 끝 임.
        # 따라서, http_url + http_query + 끝
        df_m['payload_0'] = [str(x).split(' ', maxsplit=1)[1] for x in df_m['payload_prep']]

        # url_query: payload_0에서, ' ' (공백) 첫번째를 기준으로 나누어, 1번째 값을 반환하므로, http_url ~ http_query 임.
        # 따라서, http_url + http_query
        df_m['url_query'] = [str(x).split(' ', maxsplit=1)[0] for x in df_m['payload_0']]

        except_url_query = []
        for i in df_m['payload_0']:
            if ' ' in i:
                # payload_0에서 공백이 있는 경우, http_body
                except_url_query.append(i.split(' ', maxsplit=1)[1])
                # print(except_url_query)
            else:
                except_url_query.append('')

        df_m['except_url_query'] = except_url_query
        # url_query에서 ? 가 있는 경우, 1번째 값을 반환하므로, http_url 임.
        df_m['http_url'] = [str(x).split('?', maxsplit=1)[0] for x in df_m['url_query']]

        query = []
        for i in df_m['url_query']:
            if '?' in i:
                # url_query에서, ?가 있는 경우, 2번째 값을 반환하므로, http_query 임.
                query.append('?'+i.split('?', maxsplit=1)[1])
            else:
                query.append('')
        df_m['http_query'] = query

        df_m['http_version'] = [str(x).split(' ', maxsplit=1)[0] for x in df_m['except_url_query']]

        df_m['except_version'] = [str(x).split(' ', maxsplit=1)[1] for x in df_m['except_url_query']]
        df_m['http_status'] = [str(x).split(' ', maxsplit=1)[0] for x in df_m['except_version']]

        df_m['except_status'] = [str(x).split(' ', maxsplit=1)[1] for x in df_m['except_version']]
        df_m['pkt_bytes'] = [str(x).split(' ', maxsplit=1)[0] for x in df_m['except_status']]

        df_m['except_bytes'] = [str(x).split(' ', maxsplit=1)[1] for x in df_m['except_status']]
        df_m['referer'] = [str(x).split(' ', maxsplit=1)[0] for x in df_m['except_bytes']]

        df_m['except_referer'] = [str(x).split(' ', maxsplit=1)[1] for x in df_m['except_bytes']]
        df_m['agent_etc'] = [str(x).split('"', maxsplit=1)[1] for x in df_m['except_referer']]

        df_m['user_agent'] = [str(x).split('"', maxsplit=1)[0] for x in df_m['agent_etc']]
        df_m['except_agent'] = [str(x).split('"', maxsplit=1)[1] for x in df_m['agent_etc']]

        # xforwarded_for 및 request_body 있는 경우, NGINX 임.
        if df_m.iloc[0,-1].count('"') >= 1:
        # 2022/11/14 기준 APACHE & NGINX 구분 로직 TO DO
        # 1. NGINX 처럼 APACHE, IIS에 xforwarded_for 및 request_body 필드 추가 (null 값으로)
        # 2. APACHE 이면서, SIEM RAW 필드에 'nginx' 문자열 있는 경우,  NGINX 아니면, APACHE => 이 경우, NGINX에 xforwarded_for, request_body 추가하지 않음.
        # 3. http_version 이후를, http_body 필드를 생성하여 필드 통합.

            df_m['except_agent'] = [str(x).split('"', maxsplit=1)[1] for x in df_m['except_agent']]
            df_m['xforwarded_for'] = [str(x).split('"', maxsplit=1)[0] for x in df_m['except_agent']]

            df_m['except_xforwarded'] = [str(x).split('"', maxsplit=1)[1] for x in df_m['except_agent']]
            df_m['request_body'] = [str(x).split('"', maxsplit=1)[1] for x in df_m['except_xforwarded']]

            final_df = df_m[['http_method', 'http_url', 'http_query', 'http_version', 'http_status', 'pkt_bytes', 'referer', 'user_agent', 'xforwarded_for', 'request_body']]
        
            final_np = np.where(final_df.iloc[:,:] == '', '-', final_df.iloc[:,:])
            final_df = pd.DataFrame(final_np, columns = final_df.columns)

            final_df['http_version'] = final_df['http_version'].str.replace('"', '', regex = False)
            final_df['http_status'] = final_df['http_status'].str.replace('"', '', regex = False)
            final_df['pkt_bytes'] = final_df['pkt_bytes'].str.replace('"', '', regex = False)
            final_df['referer'] = final_df['referer'].str.replace('"', '', regex = False)
            final_df['user_agent'] = final_df['user_agent'].str.replace('"', '', regex = False)
            final_df['xforwarded_for'] = final_df['xforwarded_for'].str.replace('"', '', regex = False)
            final_df['request_body'] = final_df['request_body'].str.replace('"', '', regex = False)


            # http_query 필드의 첫 글자가 '?' 인 경우, '' 처리
            if final_df.iloc[0,2].startswith('?') == True:
                final_df['http_query'] = final_df['http_query'].str[1:]

            # FLASK 적용
            flask_html = final_df.to_html(index = False, justify = 'center')
            # print(flask_df)
            # CTI 적용
            cti_json = final_df.to_json(orient = 'records')
            # print(ctf_df)
            warning_statement = 'WEB_NGINX 로그 입니다.'
        
        else:
            final_df = df_m[['http_method', 'http_url', 'http_query', 'http_version', 'http_status', 'pkt_bytes', 'referer', 'user_agent']]

            final_np = np.where(final_df.iloc[:,:] == '', '-', final_df.iloc[:,:])
            final_df = pd.DataFrame(final_np, columns = final_df.columns)

            final_df['http_version'] = final_df['http_version'].str.replace('"', '', regex = False)
            final_df['http_status'] = final_df['http_status'].str.replace('"', '', regex = False)
            final_df['pkt_bytes'] = final_df['pkt_bytes'].str.replace('"', '', regex = False)
            final_df['referer'] = final_df['referer'].str.replace('"', '', regex = False)
            final_df['user_agent'] = final_df['user_agent'].str.replace('"', '', regex = False)

            # http_query 필드의 첫 글자가 '?' 인 경우, '' 처리
            if final_df.iloc[0,2].startswith('?') == True:
                final_df['http_query'] = final_df['http_query'].str[1:]

            # FLASK 적용
            flask_html = final_df.to_html(index = False, justify = 'center')
            # print(flask_df)
            # CTI 적용
            cti_json = final_df.to_json(orient = 'records')
            # print(ctf_df)
            warning_statement = 'WEB_APACHE 로그 입니다.'


    else:

        pre_df = pd.DataFrame([raw_data_str], columns = ['payload'])
        pre_df['payload_prep'] = [str(x).split(' ', maxsplit=4)[4] for x in pre_df['payload']]
        # payload_prep 이 'http/' 부터 시작
        pre_df['start_version'] = re.findall(r'http/(.*)', pre_df.iloc[0,1], flags=re.I)
        pre_df['http_method'] = [str(x).split(' ', maxsplit=1)[0] for x in pre_df['payload_prep']]

        pre_df['start_version'] = 'HTTP/' + pre_df.iloc[0,2]

        mtd = [str(x).split(' ', maxsplit=1)[0] for x in pre_df['payload_prep']]
        for i, m in enumerate(mtd):
            if len(m) > 10 or len(m) == 1 or not m.isalpha():
                mtd[i] = ''

        method_list = ['', 'upload', 'get', 'profind', 'put', 'options', 'head', 'trace', 'connect', 'delete', 'post', 'patch']

        m_idx = []
        not_m_idx = []

        for i, m in enumerate(pre_df['http_method']):
            # if m in method_list:
            if m.lower() in method_list:
                m_idx.append(i)
            else:
                not_m_idx.append(i)


        df_m = pre_df.iloc[m_idx].reset_index(drop=True)
        df_nm = pre_df.iloc[not_m_idx].reset_index(drop=True)

        # payload_0: payload에서 ' ' (공백) 첫번째를 기준으로 나누엇, 2번째 값을 반환하므로, http_url 부터 끝 임.
        # 따라서, http_url + http_query + 끝
        df_m['payload_0'] = [str(x).split(' ', maxsplit=1)[1] for x in df_m['payload_prep']]
        # url_query: payload_0에서, ' ' (공백) 첫번째를 기준으로 나누어, 1번째 값을 반환하므로, http_url ~ http_query 임.
        # 따라서, http_url + http_query
        df_m['url_query'] = [str(x).split(' ', maxsplit=1)[0] for x in df_m['payload_0']]

        except_url_query = []
        for i in df_m['payload_0']:
            if ' ' in i:
                # payload_0에서 공백이 있는 경우, http_body
                except_url_query.append(i.split(' ', maxsplit=1)[1])
                # print(except_url_query)
            else:
                except_url_query.append('')

        df_m['except_url_query'] = except_url_query
        # url_query에서 ? 가 있는 경우, 1번째 값을 반환하므로, http_url 임.
        df_m['http_url'] = [str(x).split('?', maxsplit=1)[0] for x in df_m['url_query']]

        query = []
        for i in df_m['url_query']:
            if '?' in i:
                # url_query에서, ?가 있는 경우, 2번째 값을 반환하므로, http_query 임.
                query.append('?'+i.split('?', maxsplit=1)[1])
            else:
                query.append('')
        df_m['http_query'] = query

        df_m['http_version'] = [str(x).split(' ', maxsplit=1)[0] for x in df_m['start_version']]

        df_m['except_version'] = [str(x).split(' ', maxsplit=1)[1] for x in df_m['start_version']]
        df_m['user_agent'] = [str(x).split(' ', maxsplit=1)[0] for x in df_m['except_version']]
        
        df_m['except_agent'] = [str(x).split(' ', maxsplit=1)[1] for x in df_m['except_version']]
        df_m['referer'] = [str(x).split(' ', maxsplit=1)[0] for x in df_m['except_agent']]

        df_m['except_referer'] =  [str(x).split(' ', maxsplit=1)[1] for x in df_m['except_agent']]
        df_m['http_status'] = [str(x).split(' ', maxsplit=1)[0] for x in df_m['except_referer']]

        df_m['except_status'] = [str(x).split(' ', maxsplit=1)[1] for x in df_m['except_referer']]
        df_m['sent_bytes'] = [str(x).split(' ', maxsplit=1)[0] for x in df_m['except_status']]

        final_df = df_m[['http_method', 'http_url', 'http_query', 'http_version', 'user_agent', 'referer', 'http_status', 'sent_bytes']]
        final_np = np.where(final_df.iloc[:,:] == '', '-', final_df.iloc[:,:])
        final_df = pd.DataFrame(final_np, columns = final_df.columns)

        # http_query 필드의 첫 글자가 '?' 인 경우, '' 처리
        if final_df.iloc[0,2].startswith('?') == True:
            final_df['http_query'] = final_df['http_query'].str[1:]

        # FLASK 적용
        flask_html = final_df.to_html(index = False, justify = 'center')
        # print(flask_df)
        # CTI 적용
        cti_json = final_df.to_json(orient = 'records')
        # print(ctf_df)
        warning_statement = 'WEB_IIS 로그 입니다.'


    return render_template('WEB_payload_parsing_output.html',
                                # web_raw_data_str = request.form['web_raw_data_str'],
                                flask_html = flask_html,
                                warning_statement = warning_statement
                            )


if __name__ == '__main__':
   # cProfile.run('XAI_result()')
   app.run(host = SERVER_IP, port = PORT, debug= True )
   # app.run(host = SERVER_IP, debug= True )
   
