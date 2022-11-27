import shap
from setting import *
import torch
import scipy as sp
from transformers import pipeline, AutoTokenizer
import numpy as np
from transformers import AutoTokenizer
import os

explainer_save_path = 'YOUR SHAP EXPLAINER SAVE DIR !!!!!!!'

sql_explainer = shap.TreeExplainer(IPS_model)
text_explainer = shap.Explainer(IPS_model, train_payload_tfidf, feature_names=vectorizer.get_feature_names())
text_explainer = shap.TreeExplainer(IPS_text_model['lgbmclassifier'],
             feature_names=IPS_text_model['tfidfvectorizer'].get_feature_names_out())
total_explainer = shap.TreeExplainer(IPS_total_model)

pickle.dump(sql_explainer, open(os.path.join(explainer_save_path, 'DSS_IPS_shap_explainer.pkl'), 'wb'))
pickle.dump(text_explainer, open(os.path.join(explainer_save_path, 'DSS_IPS_text_shap_explainer.pkl'), 'wb'))
pickle.dump(total_explainer, open(os.path.join(explainer_save_path, 'DSS_IPS_total_shap_explainer.pkl'), 'wb'))
