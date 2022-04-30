

import os
import time
from DSS_IPS_preprocess import *
from setting import *

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



@app.route('/XAI_result', methods = ['POST'])
def XAI_result():

    # XAI 실행 시간
    kor_time = datetime.datetime.now()
    xai_event_time = kor_time.strftime("%Y%m%d%H%M")
    
    sql_result_total = web_UI_preprocess()
    payload_df = sql_result_total[1]
    
    expected_value = IPS_explainer.expected_value
    print('SHAP 기댓값: ', expected_value)
    # attack : shap_values[1], normal: shap_values[0]
    shap_values = IPS_explainer.shap_values(payload_df)
    
    force_plot = shap.force_plot(expected_value, shap_values[1], payload_df, link = 'logit',
                        matplotlib = False)

    force_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"


    # payload의 raw data 입력 값!
    raw_data_str = request.form['raw_data_str']

    #############################################    
    # LIME TextTabularExplainer
    # 0: normal, 1: attack
    class_names = ['Normal', 'Attack']
    
    text_explainer = LimeTextExplainer(class_names=class_names)

    pred_explainer = text_explainer.explain_instance(raw_data_str, IPS_text_model.predict_proba,
                                                num_features=10)

    pred_explainer.show_in_notebook(text=True)
    lime_text_explainer_html = pred_explainer.as_html()


    return render_template('XAI_output.html', payload_raw_data = request.form['raw_data_str'],  
                                force_html = force_html, lime_text_explainer_html = lime_text_explainer_html)






if __name__ == '__main__':
   # cProfile.run('XAI_result()')
   app.run(host = SERVER_IP, port = PORT, debug= True )
   # app.run(host = SERVER_IP, debug= True )
   