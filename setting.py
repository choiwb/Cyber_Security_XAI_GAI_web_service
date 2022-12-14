

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

