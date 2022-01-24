



from DSS_IPS_preprocess import *
from setting import *

from flask import Flask, render_template


app = Flask(__name__)
@app.route('/')
def user_input():
    return render_template('user_input.html')



@app.route('/web_UI_preprocess', methods = ['POST'])
def web_UI_preprocess():
    
    payload_df = predict_UI_sql_result()
    payload_arr = np.array(payload_df)

    return payload_arr


@app.route('/web_UI_predict', methods=['POST'])
def web_UI_predict():


    arr = np.array(web_UI_preprocess())
    # print('payload arr: ', arr)

    pred = IPS_model.predict(arr)
    pred_proba = IPS_model.predict_proba(arr)
    
    # pre_proba = ['O', 'X'] 알파벳에 따른 attack 라벨이 먼저 위치함 !!!!
    Normal_proba = int(np.round(pred_proba[:, 1], 2) * 100)
    Attack_proba = int(np.round(pred_proba[:, 0], 2) * 100)


    return render_template('server_output.html', data = [pred, Normal_proba, Attack_proba])



if __name__ == '__main__':
   app.run(host = SERVER_IP, port = PORT, debug= True )
   # app.run(host = SERVER_IP, debug= True )