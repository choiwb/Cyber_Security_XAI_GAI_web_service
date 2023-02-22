


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
plt.rcParams['font.family'] = 'NanumGothic'

import plotly.express as px
import ssdeep
import tlsh
from fuzzywuzzy import fuzz
from lime.lime_text import LimeTextExplainer
import time
import itertools

# 함수 연산시간 출력
# import cProfile


app = Flask(__name__)
@app.route('/')
def user_input():
    return render_template('user_input.html')



import multiprocessing
import openai

openai.api_key = "YOUR API KEY !!!!!!!"
ips_context_path = 'YOUR CONTEXT PATH !!!!!!!'

def load_context(file_path):
    with open(file_path, "r") as f:
        context = f.read()
    return context

def get_completion(prompt):
    completion = openai.Completion.create(
            engine='text-davinci-003',
            prompt=prompt,
            max_tokens=512,
            n=1,
            stop=None,
            temperature=0.5,
    )
    return completion

def ips_chat_gpt(raw_data_str):
    context = load_context(ips_context_path)
    
    # GPT 3.5 (text-davinci-003)는 2021년 6월 까지의 데이터로 학습된 모델 임.
    prompt_list = [
        raw_data_str + '  SQL Injection, Command Injection, XSS (Cross Site Scripting), Attempt access admin page (관리자 페이지 접근 시도), JNDI Injection, WordPress 취약점, malicious bot 총 7가지 공격 유형 중에 이 IPS 장비 payload의 경우, 어떤 공격 유형에 해당하는지 판단 근거를 작성해주세요.',
        raw_data_str + ' 2021년 6월 기준, Mitre Att&ck에서 전체 Enterprise Tactics ID 중 이 IPS 장비 payload의 경우, 적합한 Techniques ID와 간략한 설명의 경우, 한글로 작성해주세요.',
        raw_data_str + ' 이 IPS 장비 payload의 경우, 탐지할만한, Snort Rule을 작성해주세요.',
        raw_data_str + ' 이 IPS 장비 payload의 경우, 탐지할만한, Sigma Rule을 작성해주세요.',
        raw_data_str + ' 이 IPS 장비 payload의 경우, 연관될만한 CVE (Common Vulnerabilities and Exposures) 가 있으면 작성해주세요.',
        raw_data_str + ' 이 IPS 장비 payload의 경우, Cyber Kill Chain을 graph LR로 시작하는 mermaid로 작성해주세요.'
    ]

    try:
        with multiprocessing.Pool() as pool:
            completions = pool.map(get_completion, prompt_list)

        answer_strings = [c['choices'][0]['text'].strip() for c in completions]

        answer_strings = [s.replace('네, ', '').replace('아니요. ', '') for s in answer_strings]
        answer_strings[1] = answer_strings[1].replace('설명:', ' 설명:')
        answer_strings[1] = '2021년 6월 업데이트 기준 ' + answer_strings[1]

        q_and_a_df = pd.DataFrame([
            ['공격 판단 근거', answer_strings[0]],
            ['T-ID 추천', answer_strings[1]],
            ['Snort Rule 추천', answer_strings[2]],
            ['Sigma Rule 추천', answer_strings[3]],
            ['CVE 추천', answer_strings[4]]
        ], columns=['Question', 'Answer'])

        q_and_a_html = q_and_a_df.to_html(index=False, justify='center')
        q_and_a_html = q_and_a_html.replace('\\n', '')

        cy_chain_mermaid = answer_strings[5]
        # cy_chain_mermaid = cy_chain_mermaid.replace('```mermaid', '').replace('graph TD', 'graph LR').replace('graph TB', 'graph LR').replace('```', '')
        cy_chain_mermaid = cy_chain_mermaid.replace('```mermaid', '').replace('```', '')

        print(cy_chain_mermaid)

        return q_and_a_html, cy_chain_mermaid

    except:
        return '서비스 오류입니다. 다시 시도해주세요.'


###################################################################
# T-ID 분류 모델 - Tactic (14개) 별 예측 후, 상위 3개 T-ID 추출

# Mitre Att&ck 데이터 호출
mitre_attack_path = 'MITRE ATT&CK DATA PATH !!!!!'
tid_refer = pd.read_excel(mitre_attack_path, sheet_name= '세부명세')
# print(tid_refer['Tactics(ID)'].value_counts())
tactic_refer = list(tid_refer['Tactics(ID)'].value_counts().index)
# print(tactic_refer)
tactic_desc = pd.read_excel(mitre_attack_path, sheet_name= '1.Tactics(전술)')

# TFIDF 학습 키워드 호출 
tfidf_word_path = 'TRAIN SET WORD PATH !!!!!'
tfidf_word_list = os.listdir(tfidf_word_path)
tfidf_word_list.sort(key=lambda f: int(re.sub('\D', '', f)))

# Tactic 모델 호출
tactic_model_path = 'MODEL PATH !!!!!'
tactic_model_list = os.listdir(tactic_model_path)
tactic_model_list.remove('Tactic_model_TFIDF_word')
tactic_model_list.remove('mitre_attack.xlsx')
tactic_model_list.sort(key = lambda f: int(f.split('_')[1]))


for i in range(len(tfidf_word_list)):
    # tfidf_word 기반 변수명 생성
    globals()['tactic_{}_word'.format(i+1)] = pd.read_csv(os.path.join(tfidf_word_path, tfidf_word_list[i]))
    try:
        globals()['tactic_{}_model'.format(i+1)] = pickle.load(open(os.path.join(tactic_model_path, tactic_model_list[i]), 'rb'))
    except:
        print(globals()['tactic_{}_model'.format(i+1)].get_params())
        print('모델 로드 실패')
        pass
    


@app.route('/TID_TFIDF_prepro_predict_xai', methods=['POST'])
def TID_TFIDF_prepro_predict_xai():
    raw_data_str = request.form['raw_data_str']
    valid_raw_df = pd.DataFrame([raw_data_str], columns = ['total_text'])

    pred_result = pd.DataFrame(columns = ['Tactics(ID)', 'AI', 'proba', 'max_proba'])

    for i in range(len(tfidf_word_list)):

        pred_result.loc[i, 'Tactics(ID)'] = tactic_refer[i]
        # 학습 데이터의 word 및 IDF 호출
        globals()['tactic_{}_word_list'.format(i+1)] = globals()['tactic_{}_word'.format(i+1)]['word'].tolist()
        globals()['tactic_{}_idf_list'.format(i+1)] = globals()['tactic_{}_word'.format(i+1)]['IDF'].tolist()

        # valid 셋 TF 도출
        globals()['valid_{}_vectorizer'.format(i+1)] = CountVectorizer(lowercase = True, vocabulary = globals()['tactic_{}_word_list'.format(i+1)])
        globals()['valid_{}_tf_feature'.format(i+1)] = globals()['valid_{}_vectorizer'.format(i+1)].fit_transform(valid_raw_df['total_text']).toarray()
        globals()['valid_{}_tf_df'.format(i+1)] = pd.DataFrame(globals()['valid_{}_tf_feature'.format(i+1)], columns = globals()['valid_{}_vectorizer'.format(i+1)].get_feature_names_out())

        # valid 셋 TF-IDF 도출
        globals()['valid_{}_tfidf_df'.format(i+1)] = globals()['valid_{}_tf_df'.format(i+1)] * globals()['tactic_{}_idf_list'.format(i+1)]
        # print(globals()['valid_{}_tfidf_df'.format(i+1)].shape)

        # 전처리 완료된 valid 셋을 통한 Tactic 모델 별, T-IDF 에측
        globals()['tactic_{}_predict'.format(i+1)] = globals()['tactic_{}_model'.format(i+1)].predict(globals()['valid_{}_tfidf_df'.format(i+1)])
        pred_result.loc[i, 'AI'] = globals()['tactic_{}_predict'.format(i+1)][0]
        
        globals()['tactic_{}_predict_proba'.format(i+1)] = globals()['tactic_{}_model'.format(i+1)].predict_proba(globals()['valid_{}_tfidf_df'.format(i+1)])
        globals()['tactic_{}_predict_proba'.format(i+1)] = np.round(globals()['tactic_{}_predict_proba'.format(i+1)], 4)
        globals()['tactic_{}_predict_proba'.format(i+1)] = list(itertools.chain(*globals()['tactic_{}_predict_proba'.format(i+1)]))
        pred_result.loc[i, 'proba'] = globals()['tactic_{}_predict_proba'.format(i+1)]
        globals()['tactic_{}_max_proba'.format(i+1)] = max(globals()['tactic_{}_predict_proba'.format(i+1)])
        pred_result.loc[i, 'max_proba'] = globals()['tactic_{}_max_proba'.format(i+1)]
        

    pred_result['model_no'] = pred_result.index + 1
    pred_result = pred_result[['model_no', 'Tactics(ID)', 'AI', 'proba', 'max_proba']]
    # pred_result = pred_result.sort_values(by = 'max_proba', ascending = False)

    # pred_result to html
    pred_result_html = pred_result.to_html(index=False, justify='center')

    # 각 Tactic 함수 별 위 전처리 결과 통한, 예측 후, 상위 n개 T-ID 호출
    n = 5
    total_n_tid = pred_result.head(n)
    total_n_tid = total_n_tid.rename(columns = {'AI': 'Techniques(ID)'})

    #########################################################################################
    total_n_tid = total_n_tid.merge(tid_refer, how = 'left', on = ['Tactics(ID)', 'Techniques(ID)'])
    total_n_tid = total_n_tid.merge(tactic_desc, how = 'left', on = ['Tactics(ID)', 'Tactics(name)'])
    #########################################################################################
    
    total_n_tid = total_n_tid[['Tactics(ID)', 'Tactics(name)', 'Tactics 설명(간략)', 'Techniques(ID)', 'Techniques(name)', 'max_proba',
                        'Techniques 설명(번역)', 'Mitigations 설명(번역)', 'Detection 설명(번역)'
                        ]]

    total_n_tid = total_n_tid.rename(columns = {'Techniques(ID)': 'T-ID', 
                                            'Techniques(name)': 'T-ID 이름',
                                            'Tactics 설명(간략)': 'Tactic 설명',
                                            'max_proba': 'AI',
                                            'Tactics(ID)': 'Tactic',
                                            'Tactics(name)': 'Tactic 이름',
                                            'Techniques 설명(번역)': 'T-ID 설명',
                                            'Mitigations 설명(번역)': '대응 방안',
                                            'Detection 설명(번역)': '탐지 방안'})

    top_n_tid = total_n_tid.groupby(['Tactic', 'T-ID']).sample(1)
    top_n_tid = top_n_tid.sort_values(by = 'AI', ascending = False)

    top_n_tid['AI'] = top_n_tid['AI'] * 100
    # top_n_tid['Tactic AI'] 소수점 2자리까지 표현
    top_n_tid['AI'] = top_n_tid['AI'].apply(lambda x: '%.2f' % x)
    top_n_tid['AI'] = top_n_tid['AI'].astype(str)

    top_n_tid['AI'] = top_n_tid['AI'] + '%'
    top_n_tid['T-ID 설명'] = top_n_tid['T-ID 설명'].fillna('-')
    top_n_tid['대응 방안'] = top_n_tid['대응 방안'].fillna('-')
    top_n_tid['탐지 방안'] = top_n_tid['탐지 방안'].fillna('-')

    # top_n_tid to html
    top_n_tid_html = top_n_tid.to_html(index=False, justify='center')
    top_n_tid_html = top_n_tid_html.replace('\\n', '')

    return render_template('TID_multi_model_predict.html',
                                            top_n_tid_html = top_n_tid_html,
                                            )



# @app.route('/web_UI_preprocess', methods = ['GET'])
def web_UI_preprocess():
    
    payload_df = predict_UI_sql_result()

    return payload_df


@app.route('/web_UI_predict', methods=['POST'])
def web_UI_predict():
    
    # payload 입력
    payload_input = request.form['raw_data_str']

    ############################################
    # payload hash 변환 (ssdeep 이용)
    payload_hash = ssdeep.hash(payload_input)
    # payload has 변환 (tlsh 이용)
    payload_binary = payload_input.encode('utf-8')
    payload_hash = tlsh.hash(payload_binary)
    ############################################

    # payload 입력 시간
    kor_time = datetime.datetime.now()
    db_event_time = kor_time.strftime("%Y%m%d%H%M")

    sql_result_total = web_UI_preprocess() 

    payload_df = sql_result_total[1]
    payload_arr = np.array(payload_df)

    pred = IPS_total_model.predict(payload_arr)

    '''
    if pred == 1:
        db_ai = 'Anomalies'
    else:
        db_ai = 'Normal'
    '''
    pred_proba = IPS_total_model.predict_proba(payload_arr)
    Normal_proba = int(np.round(pred_proba[:, 0], 2) * 100)
    Anomalies_proba = int(np.round(pred_proba[:, 1], 2) * 100)
    
    db_proba = float(np.round(pred_proba[:, 1], 2))
 
    cur = conn.cursor()

    db_total_cols = 8 # event_time, payload_input, payload_hash, feature_names, feature_values, ai, proba, user_opinion
    db_input_str = ','.join(['%s'] * db_total_cols)

    insert_query = '''insert into ips.payload_predict_2
                  values ({});
                 '''.format(db_input_str)    
    
    # payload_arr = payload_arr.T # payload_arr Transpose 

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

    insert_record = event_time_list + payload_raw_list + payload_hash_list + feature_list + feature_value_list + ai_list + proba_list + opinion_list

    print(len(insert_record))

    cur.execute(insert_query, insert_record)
    conn.commit()


    # 입력된 payload 와 동일한 block size의 similar_df와 유사도 비교
    # 동일한 block size가 없을 경우, 유사도 0 출력!
    bs_str = payload_hash.split(':')[0]
    print('입력된 payload의 block size: ', bs_str)
    # tlsh.hash의 header (첫 3 바이트) 단, hash의 처음 'T1' 부분은 제외
    bs_str = payload_hash[2:8]
    print('입력된 payload의 block size: ', bs_str)
    

    # ssdeep & tlsh 기반 payload 유사도 측정 logic
    # select_query = '''select * from ips.payload_predict_2'''
    select_query = '''select payload_input, payload_hash, ai, proba from ips.payload_predict_2
                                      where split_part(payload_hash, ':', 1) = '{}';
                                      '''.format(bs_str)

    
    payload_predict_db = psql.read_sql(select_query, conn)

    # 입력된 payload row 제외 필요!
    block_df = payload_predict_db[['payload_input', 'payload_hash', 'ai', 'proba']].iloc[0:-1, :]
    print('동일한 block size의 payload 테이블 크기: ', block_df.shape)

    # tlsh.hash가 'TNULL' 인 경우는 제외함.
    similar_df = similar_df[similar_df['payload_hash'] != 'TNULL']
    print('TNULL 제외한 payload 테이블 크기: ', similar_df.shape)            

    # tlsh.hash = 'T1'으로 시작하는 2글자 제외
    similar_df['payload_hash'] = similar_df['payload_hash'].str[2:]


    if block_df.shape[0] != 0:

        # fuzz.ratio
        block_df['fuzz_total'] = 0
        # fuzz.partial_ratio
        block_df['fuzz_part'] = 0
        # ssdeep.compare
        block_df['compare'] = 0
        # tlsh.diff
        block_df['tlsh_diff'] = 0

        # payload DB hash 값 들과 입력된 payload hash 값 비교 및 유사도 측정
        block_df['fuzz_total'] = block_df.apply(lambda x: fuzz.ratio(payload_hash, x['payload_hash']), axis = 1)
        block_df['fuzz_part'] = block_df.apply(lambda x: fuzz.partial_ratio(payload_hash, x['payload_hash']), axis = 1)
        block_df['compare'] = block_df.apply(lambda x: ssdeep.compare(payload_hash, x['payload_hash']), axis = 1)
        block_df['tlsh_diff'] = block_df.apply(lambda x: tlsh.diff(payload_hash, x['payload_hash']), axis = 1)

        # 입력된 payload의 최대 fuzz total 유사도
        max_fuzz_total = max(block_df['fuzz_total'])
        print("입력된 payload의 최대 fuzz total 유사도 : ", max_fuzz_total)

        # 최대 유사도 기준 상위 10개 payload 추출
        top10_df = block_df.sort_values(by = 'fuzz_total', ascending = False)
        print('fuzz.ratio 기준 상위 10개 payload 추출')
        print(top10_df)

        # 입력된 payload의 최대 fuzz total 유사도 payload db에서 선택
        max_fuzz_total_payload_df = block_df[block_df['fuzz_total'] == max_fuzz_total]
        max_fuzz_total_payload_df = max_fuzz_total_payload_df.drop_duplicates(subset = ['payload_input', 'payload_hash', 'fuzz_total'])
        max_fuzz_total_payload = max_fuzz_total_payload_df.iloc[0,0]
        print("입력된 payload의 최대 fuzz total 유사도 payload : ", max_fuzz_total_payload)
        max_fuzz_total_payload_ai = max_fuzz_total_payload_df.iloc[0,2]
        print("입력된 payload의 최대 fuzz total 유사도 payload 예측 라벨 : ", max_fuzz_total_payload_ai)
        max_fuzz_total_payload_proba = max_fuzz_total_payload_df.iloc[0,3]
        print("입력된 payload의 최대 fuzz total 유사도 payload 예측 확률 : ", max_fuzz_total_payload_proba)


        # 입력된 payload의 최대 fuzz part 유사도
        max_fuzz_part = max(block_df['fuzz_part'])
        print("입력된 payload의 최대 fuzz part 유사도 : ", max_fuzz_part)

        # 입력된 payload의 최대 fuzz total 유사도 payload db에서 선택
        max_fuzz_part_payload_df = block_df[block_df['fuzz_part'] == max_fuzz_part]
        max_fuzz_part_payload_df = max_fuzz_part_payload_df.drop_duplicates(subset = ['payload_input', 'payload_hash', 'fuzz_part'])
        max_fuzz_part_payload = max_fuzz_part_payload_df.iloc[0,0]
        print("입력된 payload의 최대 fuzz part 유사도 payload : ", max_fuzz_part_payload)
        max_fuzz_part_payload_ai = max_fuzz_part_payload_df.iloc[0,2]
        print("입력된 payload의 최대 fuzz part 유사도 payload 예측 라벨 : ", max_fuzz_part_payload_ai)
        max_fuzz_part_payload_proba = max_fuzz_part_payload_df.iloc[0,3]
        print("입력된 payload의 최대 fuzz part 유사도 payload 예측 확률 : ", max_fuzz_part_payload_proba)


        # 입력된 payload의 최대 compare 유사도
        max_compare = max(block_df['compare'])
        print("입력된 payload의 최대 compare 유사도 : ", max_compare)         

        # 입력된 payload의 최대 compare 유사도 payload db에서 선택
        max_compare_payload_df = block_df[block_df['compare'] == max_compare]
        max_compare_payload_df = max_compare_payload_df.drop_duplicates(subset = ['payload_input', 'payload_hash', 'compare'])
        max_compare_payload = max_compare_payload_df.iloc[0,0]
        print("입력된 payload의 최대 compare 유사도 payload : ", max_compare_payload)
        max_compare_payload_ai = max_compare_payload_df.iloc[0,2]
        print("입력된 payload의 최대 compare 유사도 payload 예측 라벨 : ", max_compare_payload_ai)
        max_compare_payload_proba = max_compare_payload_df.iloc[0,3]
        print("입력된 payload의 최대 compare 유사도 payload 예측 확률 : ", max_compare_payload_proba)


        # 입력된 payload의 최소 tlsh.diff 유사도
        min_diff = min(block_df['tlsh_diff'])
        print("입력된 payload의 최소 tlsh.diff 유사도 : ", min_diff)         

        # 입력된 payload의 최소 tlsh.diff 유사도 payload db에서 선택
        min_diff_payload_df = block_df[block_df['tlsh_diff'] == min_diff]
        min_diff_payload_df = min_diff_payload_df.drop_duplicates(subset = ['payload_input', 'payload_hash', 'compare'])
        min_diff_payload = min_diff_payload_df.iloc[0,0]
        print("입력된 payload의 최소 tlsh.diff 유사도 payload : ", min_diff_payload)
        min_diff_payload_ai = min_diff_payload_df.iloc[0,2]
        print("입력된 payload의 최소 tlsh.diff 유사도 payload 예측 라벨 : ", min_diff_payload_ai)
        min_diff_payload__proba = min_diff_payload_df.iloc[0,3]
        print("입력된 payload의 최소 tlsh.diff 유사도 payload 예측 확률 : ", min_diff_payload__proba)


        # 입력된 payload의 유사도 측정 검증 (fuzz_ratio, fuzz_partial_ratio, ssdeep_compare, tlsh_diff)
        fuzz_ratio = fuzz.ratio(payload_hash, payload_hash)
        fuzz_part_ratio = fuzz.partial_ratio(payload_hash, payload_hash)
        ssdeep_compare = ssdeep.compare(payload_hash, payload_hash)
        tlsh_diff = tlsh.diff(payload_hash, payload_hash)

        if fuzz_ratio == 100 & fuzz_part_ratio == 100 & ssdeep_compare == 100 & tlsh_diff == 0:
            print('입력된 payload의 자기 유사성이 4가지 유사도 측정 방법 모두 100 임.')
        else:
            print('입력된 payload의 자기 유사성이 4가지 유사도 측정 방법 따라 다름 !!!!!!!')
    else:
        print('입력된 payload와 동일한 block size가 DB에 없어서 유사도 = 0')
    
    
    return render_template('server_output.html', data = [pred, Normal_proba, Anomalies_proba])


# logit (log odds) 형태를 확률로 변환
def shap_logit(x):
    logit_result = 1 / (1 + np.exp(-x))
    return logit_result

signature_list = ['/etc/passwd', 'password=admin', 'information_schema', 'xp_cmdshell', '<script']
# 탐지 패턴 소문자화
signature_list = [x.lower() for x in signature_list]

method_list = ['IGLOO-UD-File Downloading Vulnerability-1(/etc/passwd)', 'IGLOO-UD-WeakIDPasswd-1(password=admin)', 'IGLOO information_schema', 'IGLOO xp_cmdshell', 'IGLOO script']

# ai_list에 element 안에 '(.*?)'가 포함되어 있는 경우, '(.*?)' 기준으로 split 후, 리스트에 추가
first_ai_list = [x.split('(.*?)')[0] for x in ai_field if '(.*?)' in x]
end_ai_list = [x.split('(.*?)')[1] for x in ai_field if '(.*?)' in x]
except_ai_list = [x.replace('[\\.]', '.') for x in ai_field]
# ai_list의 element 안에 ('*?)' 가 2번 포함되어 있는 경우, 2번째 '(.*?)' 기준으로 split 후, 리스트에 추가
two_ai_list = [x.split('(.*?)')[2] for x in ai_field if x.count('(.*?)') == 2]
ai_list_split = first_ai_list + end_ai_list + ai_field + except_ai_list

# ai_list_split 안에 중복되는 element 가 있는 경우, 단일 처리
ai_list_split = list(set(ai_list_split))

# ai_list_split 안에 '(.*?' 나, '[\\.]' 가 포함되어 있는 경우, 제거
ai_list_split = [x for x in ai_list_split if '(.*?)' not in x]
ai_list_split = [x for x in ai_list_split if '[\\.]' not in x]

# print(ai_list_split)
# print(len(ai_list_split))


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
    # text = re.sub("(" + "|".join(ai_field) + ")", replacement_2, text, flags=re.I)

    # ai_field에서 cmd 제외
    not_cmd_field = [i for i in ai_field if i not in cmd]
    text = re.sub("(" + "|".join(not_cmd_field) + ")", replacement_2, text, flags=re.I)

    # test.split('HTTP/1.')[0]에 cmd가 있는 경우, highlight 처리
    if 'HTTP/1.' in text and text.count('HTTP/1.') == 1:
        text = re.sub("(" + "|".join(cmd) + ")", replacement_2, text.split('HTTP/1.')[0], flags=re.I) + 'HTTP/1.' + text.split('HTTP/1.')[1]

    regex = re.compile('\x1b\[103m(.*?)\x1b\[49m')
    # regex_2 = re.compile('\x1b\[91m(.*?)\x1b\[39m')

    # regex = re.compile('\033[103m(.*?)\033[49m')
    matches = [regex.match(text[i:]) for i in range(len(text))] 
    sig_pattern_prep = [m.group(0) for m in matches if m] 

    # matches_2 = [regex_2.match(text[i:]) for i in range(len(text))] 
    # ai_pattern_prep = [m.group(0) for m in matches_2 if m] 

    sig_pattern = [re.sub(r'\x1b\[103m|\x1b\[49m', '', i) for i in sig_pattern_prep]
    sig_pattern = [re.sub(r'\x1b\[91m|\x1b\[39m', '', i) for i in sig_pattern]
    # sig_pattern = [re.sub(r'\033[103m|\033[49m', '', i) for i in sig_pattern_prep]


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


# 학습 데이터 TF-IDF 호출
train_word_idf = pd.read_csv('train_word_idf.csv')
# train_word_idf = train_word_idf.sort_values(by = 'idf', descending = True)
# train_word_idf = train_word_idf.reset_index(drop = True)
# train_word_idf['order'] = train_word_idf.index + 1
# print(train_word_idf.head(10))


@app.route('/XAI_result', methods = ['POST'])
def XAI_result():

    xai_shap_save_path = 'SHAP Explainer path !!!!!'

   # payload의 raw data 입력 값!
    raw_data_str = request.form['raw_data_str']

    # XAI 실행 시간
    kor_time = datetime.datetime.now()
    xai_event_time = kor_time.strftime("%Y%m%d%H%M")
    
    payload_df = web_UI_preprocess()
    payload_arr = np.array(payload_df)

    IPS_total_explainer = pickle.load(open(IPS_total_explainer_path, 'rb'))
    # docker의 경우, explainer 직접 생성 필요 !!!!!
    # IPS_total_explainer = shap.TreeExplainer(IPS_total_model)

    expected_value_sql = IPS_total_explainer.expected_value
    expected_value_sql = np.array(expected_value_sql)
    expected_value_sql_logit = shap_logit(expected_value_sql)
    print('sql SHAP 기댓값 (logit 적용 함): ', expected_value_sql_logit)
    expected_value_sql_logit = expected_value_sql_logit[0]
    expected_value_sql_logit = np.round(expected_value_sql_logit, 4) * 100
    
    # anomalies : shap_values[1], normal: shap_values[0]
    shap_values_sql = IPS_total_explainer.shap_values(payload_arr)
    shap_values_sql = np.array(shap_values_sql)

    # shap_values_sql[1] 이 0 이상인 경우, 공격, 미만인 경우, 정상으로 판단
    shap_values_sql_direction = np.where(shap_values_sql[1] >= 0, '공격', '정상')
    print(shap_values_sql_direction)
    shap_values_sql_2 = np.abs(shap_values_sql[1]).mean(0)
    # shap_values_sql_2 합계 도출
    shap_values_sql_2_sum = np.sum(shap_values_sql_2)
    # print(shap_values_sql_2_sum)
    # shap_values_sql_2 합계를 기준으로 shap_values_sql_2의 비율 도출
    shap_values_sql_2_ratio = shap_values_sql_2 / shap_values_sql_2_sum
    shap_values_sql_2_ratio = np.round(shap_values_sql_2_ratio, 4)
    print(shap_values_sql_2_ratio)
    shap_values_sql_2_ratio_sum = np.sum(shap_values_sql_2_ratio)
    # print(shap_values_sql_2_ratio_sum)



    # shap_values_sql_logit = shap_logit(shap_values_sql)
    shap_values_sql_logit = shap_logit(shap_values_sql[1])

    print('sql SHAP values (logit 적용 함): ', shap_values_sql_logit)

    # anomalies 방향 shap_values 값
    # mean_shap_feature_values = pd.DataFrame(shap_values_sql[1], 
    #        columns=payload_df.columns).abs().mean(axis=0).sort_values(ascending=False)

    # mean_shap_values = np.abs(shap_values).mean(0)
    # mean_shap_values = np.abs(shap_values[1]).mean(0)
    # mean_shap_values = np.abs(shap_values_sql_logit).mean(0)
    # 예측 데이터는 1건이므로, 반드시 평균을 구할 필요가 없음 !!!!!

    # mean_shap_value_df = pd.DataFrame(list(zip(payload_df.columns, mean_shap_values)),
    #                               columns=['피처 명','피처 중요도'])

    shap_values_sql_direction = np.array(shap_values_sql_direction).flatten()
    mean_shap_value_df = pd.DataFrame(list(zip(payload_df.columns, shap_values_sql_2_ratio, shap_values_sql_direction)),
                                   columns=['피처 명','피처 중요도', 'AI 예측 방향'])

    
    pred = IPS_total_model.predict(payload_arr)
    if pred == 1:
        db_ai = '공격'
    else:
        db_ai = '정상'

    proba = IPS_total_model.predict_proba(payload_arr)
    attack_proba = int(np.round(proba[:, 1], 2) * 100)
    
    train_mean_df = pd.DataFrame([['모델 평균', expected_value_sql_logit, '기준'], ['예측', attack_proba, attack_proba - expected_value_sql_logit]], 
                        columns = ['모델 평균/예측', '위험도(%)', '위험도(%) 증감'])
    train_mean_df['위험도(%) 증감'][1] = np.round(train_mean_df['위험도(%) 증감'][1], 2)

    if train_mean_df['위험도(%) 증감'][1] < 0:
        train_mean_df['위험도(%) 증감'][1] = train_mean_df['위험도(%) 증감'][1]
    else:
        train_mean_df['위험도(%) 증감'] = train_mean_df['위험도(%) 증감'].astype(str)
        train_mean_df['위험도(%) 증감'][1] = '+' +  train_mean_df['위험도(%) 증감'][1]
   

   ################################################################
    # expected_value_sql_logit 기반 plotly bar chart 생성 !!!! (기준 100%)
    
    train_mean_proba_plot = px.bar(train_mean_df, x = '위험도(%)',  y = '모델 평균/예측',  
                                        orientation = 'h',
                                        text = '위험도(%)',
                                        hover_data = {'모델 평균/예측': True, '위험도(%)': True, '위험도(%) 증감': True},
                                        color = '모델 평균/예측', 
                                        color_discrete_map = {'모델 평균': '#0000FF', '예측': '#FF0000'},
                                        template = 'plotly_white')

    train_mean_proba_plot.update_layout(xaxis_fixedrange=True, yaxis_fixedrange=True,   
                        legend_itemclick = False, legend_itemdoubleclick = False,
                        showlegend = False,
                        title_text='모델 /예측 위험도', title_x=0.5,
                        yaxis_title = None,
                        # xaxis_title = None,
                        width = 900,
                        height = 250
                        )
    
    train_mean_proba_html = train_mean_proba_plot.to_html(full_html=False, include_plotlyjs=True,
                            config = {'displaylogo': False,
                            'modeBarButtonsToRemove': ['zoom', 'pan', 'zoomin', 'zoomout', 'autoscale', 'select2d', 'lasso2d',
                            'resetScale2d', 'toImage']
                            }
                            )
    
    train_mean_pred_comment = 'AI 예측 위험도는 모델 평균 (%.2f%%)에 비해 %s%%인 %d%%로 %s 예측 합니다.' % (expected_value_sql_logit, train_mean_df['위험도(%) 증감'][1], attack_proba, db_ai)
    
    '''
    if db_ai == '공격':
        mean_shap_value_df.sort_values(by=['피처 중요도'],
                                    ascending=False, inplace=True)
    else:
        mean_shap_value_df.sort_values(by=['피처 중요도'],
                                    ascending=True, inplace=True)
    '''
    
    # mean_shap_value_df 의 피처 중요도를 기준으로 내림차순 정렬
    mean_shap_value_df = mean_shap_value_df.sort_values(by=['피처 중요도'], ascending = False)
    top10_shap_values = mean_shap_value_df.iloc[0:10, :]
    top10_shap_values = top10_shap_values.reset_index(drop = True)

    top10_shap_values['순위'] = top10_shap_values.index + 1

    # 피처 설명 테이블과 join
    top10_shap_values = pd.merge(top10_shap_values, ips_feature_df, how = 'left', on = '피처 명')
    top10_shap_values = top10_shap_values[['순위', '피처 명', '피처 설명', '피처 중요도', 'AI 예측 방향']]

    payload_df_t = payload_df.T
    payload_df_t.columns = ['피처 값']
    # payload_df_t에 피처 명 컬럼 추가
    payload_df_t['피처 명'] = payload_df_t.index
    top10_shap_values = pd.merge(top10_shap_values, payload_df_t, how = 'left', on = '피처 명')
    top10_shap_values = top10_shap_values[['순위', '피처 명', '피처 설명', '피처 값', '피처 중요도', 'AI 예측 방향']]

    top10_shap_values['피처 설명'] = top10_shap_values['피처 설명'].fillna('payload에서 TF-IDF 기반 추출된 키워드에 대한 표현')

    ##################################################
    # 학습 데이터 기반 피처 중요도 요약 (상위 3개 피처)
    ##################################################
        
    first_feature = top10_shap_values.iloc[0, 1]
    first_fv = top10_shap_values.iloc[0, 3]
    second_feature = top10_shap_values.iloc[1, 1]
    second_fv = top10_shap_values.iloc[1, 3]
    third_feature = top10_shap_values.iloc[2, 1]
    third_fv = top10_shap_values.iloc[2, 3]

    '''
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

            first_statement = '%s 가 %s 하였고, 학습 데이터에서 해당 피처 값의 라벨 비율은 공격: %.2f%% 정상: %.2f%% 입니다.' %(first_feature, first_fv_result, first_fv_df_anomalies_ratio, first_fv_df_normal_ratio)
        else:
            if first_fv >= 2:
                first_statement = '로그 전송 이벤트가 %d건 이므로, 2건 이상 이어서 정상 입니다.' % first_fv
            else:
                first_statement = '로그 전송 이벤트가 %d건 입니다.' % first_fv       

    else:
        if first_fv >  0:
            first_word = first_feature[8:]

            ################################
            first_idf = train_word_idf[train_word_idf['word'] == first_word]
            first_idf_value = first_idf.iloc[0,2]
            first_order = first_idf.iloc[0,0]
            first_predict_tf = first_fv / first_idf_value
            first_predict_tf = round(first_predict_tf)
            ################################

            first_fv_df = ips_training_data[ips_training_data[first_feature] > 0]
            first_fv_df_anomalies = first_fv_df[first_fv_df['label'] == 1]
            first_fv_df_anomalies_ratio = first_fv_df_anomalies.shape[0] / first_fv_df.shape[0]
            first_fv_df_anomalies_ratio = round(first_fv_df_anomalies_ratio * 100, 2)
            first_fv_df_normal_ratio = 100 - first_fv_df_anomalies_ratio

            first_statement = '%s 키워드가 %d번 등장하였고, 학습 데이터에서 170개 키워드 중에 %s 번째로 IDF 값이 높았으며, 해당 키워드가 1번 이상 등장한 경우, 공격: %.2f%% 정상: %.2f%% 입니다.' %(first_word, first_predict_tf, first_order, first_fv_df_anomalies_ratio, first_fv_df_normal_ratio)
        else:
            first_word = first_feature[8:]

            ################################
            first_idf = train_word_idf[train_word_idf['word'] == first_word]
            first_order = first_idf.iloc[0,0]
            ################################

            first_word = first_feature[8:]
            first_fv_df = ips_training_data[ips_training_data[first_feature] == 0]
            first_fv_df_anomalies = first_fv_df[first_fv_df['label'] == 1]
            first_fv_df_anomalies_ratio = first_fv_df_anomalies.shape[0] / first_fv_df.shape[0]
            first_fv_df_anomalies_ratio = round(first_fv_df_anomalies_ratio * 100, 2)
            first_fv_df_normal_ratio = 100 - first_fv_df_anomalies_ratio
            
            first_statement = '%s 키워드가 등장하지 않았고, 학습 데이터에서 170개 키워드 중에 %s 번째로 IDF 값이 높았으며, 해당 키워드가 등장하지 않은 경우, 공격: %.2f%% 정상: %.2f%% 입니다.' %(first_word, first_order, first_fv_df_anomalies_ratio, first_fv_df_normal_ratio)


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

            second_statement = '%s 가 %s 하였고, 학습 데이터에서 해당 피처 값의 라벨 비율은 공격: %.2f%% 정상: %.2f%% 입니다.' %(second_feature, second_fv_result, second_fv_df_anomalies_ratio, second_fv_df_normal_ratio)
        else:
            if second_fv >= 2:
                second_statement = '로그 전송 이벤트가 %d건 이므로, 2건 이상 이어서 정상 입니다.' % second_fv
            else:
                second_statement = '로그 전송 이벤트가 %d건 입니다.' % second_fv        

    else:
        if second_fv > 0:
            second_word = second_feature[8:]

            ################################
            second_idf = train_word_idf[train_word_idf['word'] == second_word]
            second_idf_value = second_idf.iloc[0,2]
            second_order = second_idf.iloc[0,0]
            second_predict_tf = second_fv / second_idf_value
            second_predict_tf = round(second_predict_tf)
            ################################

            second_fv_df = ips_training_data[ips_training_data[second_feature] > 0]
            second_fv_df_anomalies = second_fv_df[second_fv_df['label'] == 1]
            second_fv_df_anomalies_ratio = second_fv_df_anomalies.shape[0] / second_fv_df.shape[0]
            second_fv_df_anomalies_ratio = round(second_fv_df_anomalies_ratio * 100, 2)
            second_fv_df_normal_ratio = 100 - second_fv_df_anomalies_ratio

            second_statement = '%s 키워드가 %d번 등장하였고, 학습 데이터에서 170개 키워드 중에 %s 번째로 IDF 값이 높았으며, 해당 키워드가 1번 이상 등장한 경우, 공격: %.2f%% 정상: %.2f%% 입니다.' %(second_word, second_predict_tf, second_order, second_fv_df_anomalies_ratio, second_fv_df_normal_ratio)
        else:
            second_word = second_feature[8:]
            
            ################################
            second_idf = train_word_idf[train_word_idf['word'] == second_word]
            second_order = second_idf.iloc[0,0]
            ################################

            second_fv_df = ips_training_data[ips_training_data[second_feature] == 0]
            second_fv_df_anomalies = second_fv_df[second_fv_df['label'] == 1]
            second_fv_df_anomalies_ratio = second_fv_df_anomalies.shape[0] / second_fv_df.shape[0]
            second_fv_df_anomalies_ratio = round(second_fv_df_anomalies_ratio * 100, 2)
            second_fv_df_normal_ratio = 100 - second_fv_df_anomalies_ratio
            
            second_statement = '%s 키워드가 등장하지 않았고, 학습 데이터에서 170개 키워드 중에 %s 번째로 IDF 값이 높았으며, 해당 키워드가 등장하지 않은 경우, 공격: %.2f%% 정상: %.2f%% 입니다.' %(second_word, second_order, second_fv_df_anomalies_ratio, second_fv_df_normal_ratio)


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

            third_statement = '%s 가 %s 하였고, 학습 데이터에서 해당 피처 값의 라벨 비율은 공격: %.2f%% 정상: %.2f%% 입니다.' %(third_feature, third_fv_result, third_fv_df_anomalies_ratio, third_fv_df_normal_ratio)
        else:
            if third_fv >= 2:
                third_statement = '로그 전송 이벤트가 %d건 이므로, 2건 이상 이어서 정상 입니다.' % third_fv
            else:
                third_statement = '로그 전송 이벤트가 %d건 입니다.' % third_fv  

    else:
        if third_fv > 0:
            third_word = third_feature[8:]
            
            ################################
            third_idf = train_word_idf[train_word_idf['word'] == third_word]
            third_idf_value = third_idf.iloc[0,2]
            third_order = third_idf.iloc[0,0]
            third_predict_tf = third_fv / third_idf_value
            third_predict_tf = round(third_predict_tf)
            ################################

            third_fv_df = ips_training_data[ips_training_data[third_feature] > 0]
            third_fv_df_anomalies = third_fv_df[third_fv_df['label'] == 1]
            third_fv_df_anomalies_ratio = third_fv_df_anomalies.shape[0] / third_fv_df.shape[0]
            third_fv_df_anomalies_ratio = round(third_fv_df_anomalies_ratio * 100, 2)
            third_fv_df_normal_ratio = 100 - third_fv_df_anomalies_ratio

            third_statement = '%s 키워드가 %d번 등장하였고, 학습 데이터에서 170개 키워드 중에 %s 번째로 IDF 값이 높았으며, 해당 키워드가 1번 이상 등장한 경우, 공격: %.2f%% 정상: %.2f%% 입니다.' %(third_word, third_predict_tf, third_order, third_fv_df_anomalies_ratio, third_fv_df_normal_ratio)
        else:
            third_word = third_feature[8:]

            ################################
            third_idf = train_word_idf[train_word_idf['word'] == third_word]
            third_order = third_idf.iloc[0,0]
            ################################

            third_fv_df = ips_training_data[ips_training_data[third_feature] == 0]
            third_fv_df_anomalies = third_fv_df[third_fv_df['label'] == 1]
            third_fv_df_anomalies_ratio = third_fv_df_anomalies.shape[0] / third_fv_df.shape[0]
            third_fv_df_anomalies_ratio = round(third_fv_df_anomalies_ratio * 100, 2)
            third_fv_df_normal_ratio = 100 - third_fv_df_anomalies_ratio

            third_statement = '%s 키워드가 등장하지 않았고, 학습 데이터에서 170개 키워드 중에 %s 번째로 IDF 값이 높았으며, 해당 키워드가 등장하지 않은 경우, 공격: %.2f%% 정상: %.2f%% 입니다.' %(third_word, third_order, third_fv_df_anomalies_ratio, third_fv_df_normal_ratio)
    '''


    # 소수점 4째 자리까지 표현
    top10_shap_values['피처 값'] = top10_shap_values['피처 값'].apply(lambda x: round(x, 4))
    top10_shap_values['피처 값'] = top10_shap_values['피처 값'].astype(str)

    # 피처 명이 ips_로 시작하는 경우 또는 피처 값이 0인 경우, 피처 값은 정수로 표현
    top10_shap_values['피처 값'] = top10_shap_values.apply(lambda x: x['피처 값'].split('.')[0]
                                    if x['피처 명'].startswith('ips_') or x['피처 값'] == '0.0'
                                    else x['피처 값'], 
                                    axis = 1)



    # top10_shap_values['피처 명'] 에서 'ips_00001_' 제거
    top10_shap_values['피처 명'] = top10_shap_values['피처 명'].apply(lambda x: x[10:] if x.startswith('ips_00001_') else x)

    top10_shap_values = top10_shap_values.drop('순위', axis=1)

    # top10_shap_values의 피처 중요도 합계 
    top10_shap_values_sum = top10_shap_values['피처 중요도'].sum()
    top10_shap_values_sum_etc = 1 - top10_shap_values_sum
    etc_df = pd.DataFrame([['기타', '상위 10개 이외 피처', '-', top10_shap_values_sum_etc, '기타']], columns = ['피처 명', '피처 설명', '피처 값', '피처 중요도', 'AI 예측 방향'])
    top10_shap_values = pd.concat([top10_shap_values, etc_df], axis=0)
    top10_shap_values = top10_shap_values.sort_values(by='피처 중요도', ascending=False)
    top10_shap_values = top10_shap_values.reset_index(drop = True)

    top10_shap_values['순위'] = top10_shap_values.index + 1
    top10_shap_values  = top10_shap_values[['순위', '피처 명', '피처 설명', '피처 값', '피처 중요도', 'AI 예측 방향']]
    top10_shap_values['피처 중요도'] = top10_shap_values['피처 중요도'].apply(lambda x: round(x, 4))

    # 보안 시그니처 패턴 리스트 highlight
    sig_ai_pattern, sig_df = highlight_text(raw_data_str, signature_list, ai_field)
    # print(sig_ai_pattern)

    # 위 12개 피처가 payload의 AI 탐지되면 추출 !!!!!!!
    # sig_ai_pattern 에서 추출 및 상위 10개 피처에 대해서만 적용
    # sig_ai_pattern에서 \033[91m ~ \033[39m 사이 키워드 추출
    ai_detect_regex = r'\x1b\[91m(.*?)\x1b\[39m'
    ai_detect_list = re.findall(ai_detect_regex, sig_ai_pattern)
    ai_detect_list = [re.sub(r'\x1b\[103m|\x1b\[49m', '', x) for x in ai_detect_list]


    ai_feature_list = []
    '''
    ai_feature_list.append(['payload_sql_comb_01' for x in ai_detect_list for y in sql_1 if re.search(y, x.lower())])
    ai_feature_list.append(['payload_sql_comb_02' for x in ai_detect_list for y in sql_2 if re.search(y, x.lower())])
    ai_feature_list.append(['payload_sql_comb_03' for x in ai_detect_list for y in sql_3 if re.search(y, x.lower())])
    ai_feature_list.append(['payload_log4j_comb_01' for x in ai_detect_list for y in log4j if re.search(y, x.lower())])
    ai_feature_list.append(['payload_xss_comb_01' for x in ai_detect_list for y in xss if re.search(y, x.lower())])
    ai_feature_list.append(['payload_cmd_comb_01' for x in ai_detect_list for y in cmd if re.search(y, x.lower())])
    ai_feature_list.append(['payload_wp_comb_01' for x in ai_detect_list for y in wp if re.search(y, x.lower())])
    ai_feature_list.append(['payload_word_comb_01' for x in ai_detect_list for y in word_1 if re.search(y, x.lower())])
    ai_feature_list.append(['payload_word_comb_02' for x in ai_detect_list for y in word_2 if re.search(y, x.lower())])
    ai_feature_list.append(['payload_word_comb_03' for x in ai_detect_list for y in word_3 if re.search(y, x.lower())])
    ai_feature_list.append(['payload_word_comb_04' for x in ai_detect_list for y in word_4 if re.search(y, x.lower())])
    ai_feature_list.append(['payload_useragent_comb' for x in ai_detect_list for y in user_agent if re.search(y, x.lower())])
    '''
    
    for x in ai_detect_list:
        for y in sql_1:
            if re.search(y, x.lower()):
                ai_feature_list.append(['payload_sql_comb_01'])
                break
        for y in sql_2:
            if re.search(y, x.lower()):
                ai_feature_list.append(['payload_sql_comb_02'])
                break
        for y in sql_3:
            if re.search(y, x.lower()):
                ai_feature_list.append(['payload_sql_comb_03'])
                break
        for y in log4j:
            if re.search(y, x.lower()):
                ai_feature_list.append(['payload_log4j_comb_01'])
                break
        for y in xss:
            if re.search(y, x.lower()):
                ai_feature_list.append(['payload_xss_comb_01'])
                break
        for y in cmd:
            if re.search(y, x.lower()):
                ai_feature_list.append(['payload_cmd_comb_01'])
                break
        for y in wp:
            if re.search(y, x.lower()):
                ai_feature_list.append(['payload_wp_comb_01'])
                break
        for y in word_1:
            if re.search(y, x.lower()):
                ai_feature_list.append(['payload_word_comb_01'])
                break
        for y in word_2:
            if re.search(y, x.lower()):
                ai_feature_list.append(['payload_word_comb_02'])
                break
        for y in word_3:
            if re.search(y, x.lower()):
                ai_feature_list.append(['payload_word_comb_03'])
                break
        for y in word_4:
            if re.search(y, x.lower()):
                ai_feature_list.append(['payload_word_comb_04'])
                break
        for y in user_agent:
            if re.search(y, x.lower()):
                ai_feature_list.append(['payload_useragent_comb'])
                break

    
    # ai_feature_list = [x for x in ai_feature_list if x != []]
    ai_feature_list = list(itertools.chain(*ai_feature_list))
    
    


    # ai_feature_list, ai_detect_list 를 이용하여 2개 컬럼 기반 data frame 생성
    print(ai_detect_list)
    print(ai_feature_list)
    ai_feature_df = pd.DataFrame({'피처 명': ai_feature_list, 'AI 공격 탐지 키워드': ai_detect_list})

    # ai_feature_df['피처 명'] 중복된 행이 있다면, ',' 기준 concat
    ai_feature_df = ai_feature_df.groupby('피처 명')['AI 공격 탐지 키워드'].apply(','.join).reset_index()


    # print(ai_feature_df)
    top10_shap_values = top10_shap_values.merge(ai_feature_df, how='left', on='피처 명')
    top10_shap_values['AI 공격 탐지 키워드'] = top10_shap_values['AI 공격 탐지 키워드'].fillna('-')

    top10_shap_values['피처 중요도'] = np.round(top10_shap_values['피처 중요도'] * 100, 2)
    top10_shap_values = top10_shap_values.rename(columns = {'피처 중요도': '피처 중요도(%)'})

    # top10_shap_values의 피처 중요도 합계 
    top10_shap_values_sum = top10_shap_values['피처 중요도(%)'].sum()
    # top10_shap_values_sum_etc = 1 - top10_shap_values_sum
    # etc_df = pd.DataFrame([['기타', '상위 10개 이외 피처', '-', top10_shap_values_sum_etc, '기타']], columns = ['피처 명', '피처 설명', '피처 값', '피처 중요도', 'AI 예측 방향'])
    # top10_shap_values = pd.concat([top10_shap_values, etc_df], axis=0)
    # top10_shap_values = top10_shap_values.sort_values(by='피처 중요도', ascending=False)
    # top10_shap_values = top10_shap_values.reset_index(drop = True)


    ##################################################
    # 학습 데이터 기반 피처 중요도 요약 (상위 3개 피처)
    ##################################################

    first_feature = top10_shap_values.iloc[0, 1]
    first_fv = top10_shap_values.iloc[0, 3]
    first_word = top10_shap_values.iloc[0,-1]
    second_feature = top10_shap_values.iloc[1, 1]
    second_fv = top10_shap_values.iloc[1, 3]
    second_word = top10_shap_values.iloc[1,-1]
    third_feature = top10_shap_values.iloc[2, 1]
    third_fv = top10_shap_values.iloc[2, 3]
    third_word = top10_shap_values.iloc[2,-1]


    
    if first_feature != 'payload_whitelist':
        if first_fv == 1:
            first_fv_result = '공격 탐지'
            first_statement = '%s 가 %s 하였고 AI 탐지 키워드는 %s 입니다.'  %(first_feature, first_fv_result, first_word)
        else:
            first_fv_result = '정상 인식'
            first_statement = '%s 가 %s 하였습니다.' %(first_feature, first_fv_result)
    else:
        first_statement = '로그 전송이 총 %s건 입니다.' % first_fv       


    if second_feature != 'payload_whitelist':
        if second_fv == 1:
            second_fv_result = '공격 탐지'
            second_statement = '%s 가 %s 하였고 AI 탐지 키워드는 %s 입니다.'  %(second_feature, second_fv_result, second_word)
        else:
            second_fv_result = '정상 인식'
            second_statement = '%s 가 %s 하였습니다.' %(second_feature, second_fv_result)
    else:
        second_statement = '로그 전송이 총 %s건 입니다.' % second_fv       


    if third_feature != 'payload_whitelist':
        if third_fv == 1:
            third_fv_result = '공격 탐지'
            third_statement = '%s 가 %s 하였고 AI 탐지 키워드는 %s 입니다.'  %(third_feature, third_fv_result, third_word)
        else:
            third_fv_result = '정상 인식'
            third_statement = '%s 가 %s 하였습니다.' %(third_feature, third_fv_result)
    else:
        third_statement = '로그 전송이 총 %s건 입니다.' % third_fv       
    # top10_shap_values to html
    top10_shap_values_html = top10_shap_values.to_html(index=False, justify='center')


    # top10_shap_values to plotly                         
    # 피처 중요도에 커서 올리면 피처 설명 나오도록 표시
    # background color = white
    # 피처 중요도 기준 0.5 이상은 '공격' 미만은 '정상'
    # top10_shap_values['AI 예측 방향'] = ['공격' if x >= 0.5 else '정상' for x in top10_shap_values['피처 중요도']]

    summary_plot = px.bar(top10_shap_values, x="피처 중요도(%)", y="피처 명", 
                color = 'AI 예측 방향', color_discrete_map = {'공격': '#FF0000', '정상': '#00FF00', '기타': '#0000FF'},
                text = '피처 중요도(%)', orientation='h', hover_data = {'피처 명': False, '피처 설명': True, '피처 값': True, '피처 중요도(%)': False, 'AI 예측 방향': False,
                                                                    'AI 공격 탐지 키워드': True},
                template = 'plotly_white',
                )
    
    # 피처 중요도에 따른 sort reverse !!!!!
    # 피처 중요도 기준 내림 차순 정렬
    summary_plot.update_layout(xaxis_fixedrange=True, yaxis_fixedrange=True,
                            yaxis = dict(autorange="reversed"),
                            yaxis_categoryorder = 'total descending',
                            legend_itemclick = False, legend_itemdoubleclick = False,
                            title_text='AI 예측 피처 중요도', title_x=0.5,
                            yaxis_title = None
                            )
    
    # plotly to html and all config false
    summary_html = summary_plot.to_html(full_html=False, include_plotlyjs=True,
                                config = {'displaylogo': False,
                                'modeBarButtonsToRemove': ['zoom', 'pan', 'zoomin', 'zoomout', 'autoscale', 'select2d', 'lasso2d',
                                'resetScale2d', 'toImage']
                                }
                                )

    pie_plot = px.pie(top10_shap_values, values='피처 중요도', names='피처 명',
                                                color = 'AI 예측 방향',
                                                color_discrete_map = {'공격': '#FF0000', '정상': '#00FF00', '기타': '#0000FF'},
                                                template = 'plotly_white',
                                                custom_data = ['피처 설명', '피처 값', 'AI 예측 방향'],
                                                # hover_data = ['피처 설명', '피처 값', 'AI 예측 방향'],
                                                labels = ['피처 명']
                                                )
    
    # print(top10_shap_values.dtypes)

    # custom_data 에서 피처 설명, 피처 값, AI 예측 방향을 가져와서 ',' 기준 split 하여 표시
    pie_plot.update_traces(textposition='inside', textinfo='label+percent',
                           hovertemplate = '피처 명: %{label}<br>' +
                                            '피처 중요도: %{value:.4f}<br>' +
                                            '피처 설명: %{customdata[0][0]}<br>' +
                                            '피처 값: %{customdata[0][1]}<br>' +
                                            'AI 예측 방향: %{customdata[0][2]}<br>',    
                           hole = 0.3,
                           # hoverinfo = 'label+value'
                            )

    pie_plot.update_layout(xaxis_fixedrange=True, yaxis_fixedrange=True,
                           legend_itemclick = False, legend_itemdoubleclick = False,
                            title_text='AI 예측 피처 중요도', title_x=0.5,
                            annotations = [dict(text = '위험도: %d%%<br>%s' %(attack_proba, db_ai),
                            x = 0.5, y = 0.5, 
                            font_color = '#FF0000' if db_ai == '공격' else '#00FF00',
                            font_size = 12, showarrow = False)]
                            )

    pie_plot.update(layout_showlegend=True)

    ###################################
    # 1. 전체 피처 중 공격/정상 예측에 영향을 준 상위 10개 피처 비율은 몇 % 이다.
    pie_statement_1 = "전체 피처 중 공격/정상 예측에 영향을 준 상위 10개 피처 비율은 {:.2f}%를 차지.".format(top10_shap_values_sum * 100)
    # 2. 상위 10개 피처 중 공격 예측에 영향을 준 피처는 전체 피처 중 몇 % 이다.
    pie_statement_2 = "상위 10개 피처 중 공격 예측에 영향을 준 피처는 전체 피처 중 {:.2f}%를 차지.".format(top10_shap_values[top10_shap_values['AI 예측 방향'] == '공격']['피처 중요도'].sum() * 100)
    ###################################

    pie_html = pie_plot.to_html(full_html=False, include_plotlyjs=True,
                                config = {'displaylogo': False,
                                'modeBarButtonsToRemove': ['zoom', 'pan', 'zoomin', 'zoomout', 'autoscale', 'select2d', 'lasso2d',
                                'resetScale2d', 'toImage']
                                }
                                )   

    # higher: red, lower: green
    shap_cols = payload_df.columns.tolist()
    # payload_df.columns startswith 'ips_00001' 인 경우, ''로 변경
    shap_cols = [x.replace('ips_00001_', '') for x in shap_cols]

    # force_plot = plt.figure()
    force_plot = shap.force_plot(expected_value_sql[0], shap_values_sql[1], payload_df, link = 'logit',
                        plot_cmap = ['#FF0000', '#00FF00'],
                        feature_names = shap_cols,
                        out_names = '공격',
                        matplotlib = False)

    # force_plot의 각 피처별 proba 형태 총 100% 되도록 추출
    

    # plt.savefig('static/force_plot.png', bbox_inches = 'tight', dpi = 500)
    force_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
    
    #############################################    
    # SHAP's force plot - text feature
    
    payload_str_df = pd.DataFrame([raw_data_str], columns = ['payload'])
    payload_str = payload_str_df['payload']

    payload_test_tfidf = IPS_text_model['tfidfvectorizer'].transform(payload_str).toarray()
    IPS_text_explainer = pickle.load(open(IPS_text_explainer_path, 'rb'))
    
    # IPS_text_explainer = shap.TreeExplainer(IPS_text_model['lgbmclassifier'],
    #               feature_names=IPS_text_model['tfidfvectorizer'].get_feature_names_out())
    
    
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
    # BERT explainer 선언이 아닌, 저장 후 호출 테스트 필요 !!!!!
    #########################################################
    IPS_pytorch_bert_explainer = shap.Explainer(bert_predict, tokenizer)
    #########################################################
    bert_shap_values = IPS_pytorch_bert_explainer(bert_payload, fixed_context=1, batch_size=1)
    # print('@@@@@@@@@@@@@@@@@')
    # print(bert_shap_values)
    # print('@@@@@@@@@@@@@@@@@')
    bert_shap_end = time.time()
    dur_bert_shap = bert_shap_end - bert_shap_start
    # cpu 연산 시간: 11.23 초, mps 연산 시간: 7.46 초
    print('mps 연산 시간: %.2f (초)' %(dur_bert_shap))

    bert_token = list(bert_shap_values.data)
    # print(bert_token)
    bert_values = bert_shap_values.values
    # print(bert_values)
    bert_values_logit = shap_logit(bert_values)
    # bert_values_logit 소수점 4자리까지 표현
    bert_values_logit = np.round(bert_values_logit, 4)
    # print(bert_values_logit)

    text_html = shap.text_plot(bert_shap_values, display = False)

    # HTML 형태 payload 의 경우, 소괄호 치환 필요
    sig_ai_pattern = re.sub(r'[\\<]', r'&lt;', sig_ai_pattern)
    sig_ai_pattern = re.sub(r'[\\>]', r'&gt;', sig_ai_pattern)

    foreground_regex = r'\x1b\[91m(.*?)\x1b\[39m'
    background_regex = r'\x1b\[103m(.*?)\x1b\[49m'
    
    sig_ai_pattern = re.sub(foreground_regex, r'<font color = "red">\1</font>', sig_ai_pattern)
    sig_ai_pattern = re.sub(background_regex, r'<span style = "background-color:yellow;">\1</span>', sig_ai_pattern)
    
    sig_pattern_html = f"<head>{sig_ai_pattern}</head>"        
    sig_df_html = sig_df.to_html(index=False, justify='center')
    
    start_chat_api = time.time()
    try:
        q_and_a_html, cy_chain_mermaid = ips_chat_gpt(raw_data_str)
    except:
        q_and_a_html = '서비스 오류입니다. 다시 시도해주세요.'
        cy_chain_mermaid = '서비스 오류입니다. 다시 시도해주세요.'
    end_chat_api = time.time()
    print('Open AI 챗봇 호출 시간: %.2f (초)' %(end_chat_api - start_chat_api))


    return render_template('XAI_output.html', payload_raw_data = request.form['raw_data_str'],  
                                # expected_value_sql_logit = expected_value_sql_logit,
                                train_mean_proba_html = train_mean_proba_html,
                                train_mean_pred_comment = train_mean_pred_comment,
                                force_html = force_html,
                                summary_html = summary_html,
                                # pie_html = pie_html,
                                text_explainer_html = text_explainer_html,
                                lime_text_explainer_html = lime_text_explainer_html, 
                                text_html = text_html,
                                bert_label = bert_label,
                                bert_score = bert_score,
                                # top10_shap_values_html = top10_shap_values_html,
                                first_statement = first_statement,
                                second_statement = second_statement,
                                third_statement = third_statement,
                                summary_statement_1 = summary_statement_1,
                                summary_statement_2 = summary_statement_2,
                                sig_pattern_html = sig_pattern_html,
                                sig_df_html = sig_df_html,
                                # summary_html = summary_html,
                                q_and_a_html = q_and_a_html,
                                cy_chain_mermaid = cy_chain_mermaid
                                )


@app.route('/WAF_payload_parsing', methods = ['POST'])
def WAF_payload_parsing():
    raw_data_str = request.form['raw_data_str']

    # raw_data_str '"'로 시작하는 경우 '' 처리
    if raw_data_str[0] == '"':
        raw_data_str = raw_data_str[1:]

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
    from waitress import serve
    serve(app, host=SERVER_IP, port=PORT)
    # app.run(host = SERVER_IP, port = PORT, debug= True)

