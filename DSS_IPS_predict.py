

import os
import re
import time
from DSS_IPS_preprocess import *
from setting import *
from DSS_IPS_shap_explainer_save import *

from flask import Flask, render_template, request
import numpy as np
import shap

import datetime
import pandas.io.sql as psql

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ssdeep
from fuzzywuzzy import fuzz
from lime.lime_text import LimeTextExplainer



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

# 시그니처 패턴 및 AI 피처 하이라이트 처리 위한 리스트
ai_field = ['select(.*?)from', 'cmd', 'wget', 'password', 'from(.*?)group(.*?)by']


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
    
    expected_value_sql = IPS_total_explainer.expected_value
    expected_value_sql = np.array(expected_value_sql)
    expected_value_sql_logit = shap_logit(expected_value_sql)
    print('sql SHAP 기댓값 (logit 적용 함): ', expected_value_sql_logit)

    # anomalies : shap_values[1], normal: shap_values[0]
    shap_values_sql = IPS_total_explainer.shap_values(payload_arr)
    shap_values_sql = np.array(shap_values_sql)
    shap_values_sql_logit = shap_logit(shap_values_sql)
    print('sql SHAP values (logit 적용 함): ', shap_values_sql_logit)

    shap_mean_values = np.abs(shap_values_sql[1]).mean(0)
    mean_shap_feature_values = pd.DataFrame(list(zip(payload_df.columns, shap_mean_values)), 
            columns=['feature_names', 'shap_values'])
    mean_shap_feature_values.sort_values(by=['shap_values'],ascending=False,inplace=True)

    top10_shap_values = mean_shap_feature_values.iloc[0:10, :]
    print(top10_shap_values)
    '''logit 변환 고려 !!!!!!!!!!!!!!!!!!!'''
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

    # transformers == 4.21.3 에서만 동작 (4.23.1 에서는 에러) => 20221101 rlwns
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
    bert_shap_values = IPS_pytorch_bert_explainer(bert_payload, fixed_context=1, batch_size=1)

    text_html = shap.text_plot(bert_shap_values, display = False)
    # text_html = f"<head>{shap.getjs()}</head><body>{text_plot.html()}</body>"

    # 보안 시그니처 패턴 리스트 highlight
    # sig_pattern, sig_df = highlight_text(raw_data_str, signature_list)
    sig_ai_pattern, sig_df = highlight_text(raw_data_str, signature_list, ai_field)

    print(sig_ai_pattern)
    print(sig_df)

    # sig_pattern (specific word) in text to html
    sig_ai_pattern = re.sub(r'\x1b\[103m', r'<mark>', sig_ai_pattern)
    sig_ai_pattern = re.sub(r'\x1b\[49m', r'</mark>', sig_ai_pattern)

    sig_ai_pattern = re.sub(r'\x1b\[91m', r'<span style = "color:red;">', sig_ai_pattern)
    sig_ai_pattern = re.sub(r'\x1b\[39m', r'</span>', sig_ai_pattern)

    # sig_pattern = sig_pattern.replace('\x1b\[103m', '<mark>').replace('\x1b\[49m', '</mark>')
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
                                sig_pattern_html = sig_pattern_html,
                                sig_df_html = sig_df_html,
                                # summary_html = summary_html
                                )





if __name__ == '__main__':
   # cProfile.run('XAI_result()')
   app.run(host = SERVER_IP, port = PORT, debug= True )
   # app.run(host = SERVER_IP, debug= True )
   