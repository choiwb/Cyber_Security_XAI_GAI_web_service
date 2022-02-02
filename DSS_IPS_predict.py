


import time
from DSS_IPS_preprocess import *
from setting import *

from flask import Flask, render_template, request

import transformers
import shap

# 함수 연산시간 출력
import cProfile


app = Flask(__name__)
@app.route('/')
# @app.route('/XAI')
def user_input():
    return render_template('user_input.html')



# @app.route('/web_UI_preprocess', methods = ['GET'])
def web_UI_preprocess():
    
    payload_df = predict_UI_sql_result()
    payload_arr = np.array(payload_df)

    return payload_arr, payload_df


    
@app.route('/web_UI_predict', methods=['POST'])
def web_UI_predict():

    sql_result_total = web_UI_preprocess() 

    # payload_arr = np.array(sql_Result_total[0])
    payload_arr = np.array(sql_result_total[1])
    # print('payload arr: ', arr)

    pred = IPS_model.predict(payload_arr)
    pred_proba = IPS_model.predict_proba(payload_arr)
    
    Normal_proba = int(np.round(pred_proba[:, 0], 2) * 100)
    Attack_proba = int(np.round(pred_proba[:, 1], 2) * 100)


    return render_template('server_output.html', data = [pred, Normal_proba, Attack_proba])



@app.route('/XAI_result', methods = ['POST'])
def XAI_result():
    
    sql_result_total = web_UI_preprocess()
    payload_df = sql_result_total[1]
    
    shap_start = time.time()

    explainer = shap.TreeExplainer(IPS_model)
    shap_values = explainer.shap_values(payload_df)
    print(shap_values)

    shap_end = time.time()
    shap_total = shap_end - shap_start
    print('SHAP explainer 생성 연산 시간: %.2f (초)' %(shap_total))


    force_plot = shap.force_plot(explainer.expected_value[1], shap_values[1], payload_df, link = 'logit',
                        matplotlib = False)


    # shap_html = f"<head>{shap.getjs()}</head><body><center><h1><br> SHAP - force plot </h1><br>{force_plot.html()}</center></body>"
    force_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"


    # shap.initjs()
    # shap.force_plot(explainer.expected_value, shap_values, payload_df)

    return render_template('XAI_output.html', payload_raw_data = request.form['raw_data_str'],  
                                force_html = force_html)

    # return shap_html



if __name__ == '__main__':
   # cProfile.run('XAI_result()')
   app.run(host = SERVER_IP, port = PORT, debug= True )
   # app.run(host = SERVER_IP, debug= True )