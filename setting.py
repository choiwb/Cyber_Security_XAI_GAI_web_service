

import pickle
import psycopg2 as pg2


SERVER_IP = '127.0.0.1'
PORT = 17171 

IPS_model_path = 'YOUR OWN PRE TRAINED MODEL PATH!!!!!!!!'
IPS_explainer_path = 'YOUR OWN SHAP EXPLAINER PATH!!!!!!!!'
IPS_text_model_path = 'YOUR OWN TEXT FEATURE MODEL PATH!!!!!!!!'

IPS_model = pickle.load(open(IPS_model_path, 'rb'))
IPS_explainer = pickle.load(open(IPS_explainer_path, 'rb'))
IPS_text_model = pickle.load(open(IPS_text_model_path, 'rb'))


# PostgreSQL - Payload 예측 DB 연동
conn = pg2.connect('''host = ??  
                    dbname =  ??
                    user =  ??
                    password = ??  
                    port = ?? ''')



