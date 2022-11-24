import os
import pickle
import shap
from setting import *
import torch
from transformers import AutoTokenizer
import numpy as np
import scipy as sp

explainer_save_path = 'SHAP explainer save path !!!'

sql_explainer = shap.TreeExplainer(IPS_model)
text_explainer = shap.TreeExplainer(IPS_text_model['lightgbmclassifier'],
                feature_names=IPS_text_model['tfidfvectorizer'].get_feature_names_out())

pickle.dump(sql_explainer, open(os.path.join(explainer_save_path, 'DSS_IPS_shap_explainer.pkl'), 'wb'))
pickle.dump(text_explainer, open(os.path.join(explainer_save_path, 'DSS_IPS_text_shap_explainer.pkl'), 'wb'))


device = torch.device('mps')

model_checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

IPS_pytorch_bert_model_path = 'BERT transfer learning model PATH !!!!!!!'
bert_model = torch.load(IPS_pytorch_bert_model_path)
print('BERT 전이학습 모델 평가: ', bert_model.eval())

bert_pipe = pipeline(task = "text-classification",
                model = bert_model,
                tokenizer = tokenizer,
                device = device)


# define a prediction function
def bert_predict(x):
    tv = torch.tensor([tokenizer.encode(v, padding='max_length', max_length=256, truncation=True) for v in x]).to(device)

    # outputs = model(tv)[0].detach().cpu().numpy()
    outputs = bert_model(tv)[0].detach().cpu().numpy()

    scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
    val = sp.special.logit(scores[:,1]) # use one vs rest logit units

    return val

# build an explainer using a token masker
pytorch_bert_explainer = shap.Explainer(bert_predict, tokenizer)

pickle.dump(pytorch_bert_explainer, open(os.path.join(explainer_save_path, 'DSS_IPS_pytorch_bert_explainer.pkl'), 'wb'))
