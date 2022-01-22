


import pickle

SERVER_IP = '127.0.0.1'
PORT = 17171


IPS_model_path = '/Users/choiwb/Python_projects/이글루시큐리티_DSS_표준모델_Web_API/DSS_IPS_flask_server/saved_model/DSS_IPS_LightGBM.pkl'
IPS_model = pickle.load(open(IPS_model_path, 'rb'))