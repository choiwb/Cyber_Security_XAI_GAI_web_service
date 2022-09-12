import os
import pickle
import shap
from setting import *
import torch
import transformers
import numpy as np
import scipy as sp

explainer_save_path = 'SHAP explainer save path !!!'

sql_explainer = shap.TreeExplainer(IPS_model)
text_explainer = shap.TreeExplainer(IPS_text_model['catboostclassifier'],
                feature_names=IPS_text_model['tfidfvectorizer'].get_feature_names_out())

pickle.dump(sql_explainer, open(os.path.join(explainer_save_path, 'DSS_IPS_shap_explainer.pkl'), 'wb'))
pickle.dump(text_explainer, open(os.path.join(explainer_save_path, 'DSS_IPS_text_shap_explainer.pkl'), 'wb'))

device = torch.device('mps')


# load a BERT sentiment analysis model
# BERT 예측 속도 개선 시, tokenizer 개발
tokenizer = transformers.DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased",
                    do_lowercase=True, add_special_tokens=True, max_length=128, pad_to_max_length=True,
                    truncation=True, return_tensors='pt', num_labels = 2
                    )

model = transformers.DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english", 
)

# pickle.dump(model, open(os.path.join(explainer_save_path, 'DSS_IPS_bert_model.pkl'), 'wb'))

IPS_pytorch_bert_model_path = ''
IPS_pytorch_bert_model = pickle.load(open(IPS_pytorch_bert_model_path, 'rb'))


# logit (log odds) 형태를 확률로 변환
def shap_logit(x):
    logit_result = 1 / (1 + np.exp(-x))
    return logit_result


# define a prediction function
def bert_predict(x):
    # tv = torch.tensor([tokenizer.encode(v, padding='max_length', max_length=100, truncation=True) for v in x]).to(device)
    tv = torch.tensor([tokenizer.encode(v, padding='max_length', max_length=128, truncation=True) for v in x])

    #outputs = model(tv)[0].detach().cpu().numpy()
    outputs = IPS_pytorch_bert_model(tv)[0].detach().cpu().numpy()

    scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
    '''logit => probability 형태 변환 필요 !!!!!'''
    val = sp.special.logit(scores[:,1]) # use one vs rest logit units
    # val = shap_logit(val)
    
    return val

# build an explainer using a token masker
pytorch_bert_explainer = shap.Explainer(bert_predict, tokenizer)

'''예측 중 실시간 저장되는 이슈가 있음 !!!! 아래 코드 주석 처리 시 에러 발생 함 !!!!!'''
pickle.dump(pytorch_bert_explainer, open(os.path.join(explainer_save_path, 'DSS_IPS_pytorch_bert_explainer.pkl'), 'wb'))
