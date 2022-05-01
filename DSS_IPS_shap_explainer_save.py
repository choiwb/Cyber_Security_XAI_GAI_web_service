import os
import pickle
import shap
from setting import *


explainer_save_path = 'SHAP explainer save path !!!'

sql_explainer = shap.TreeExplainer(IPS_model)
text_explainer = shap.TreeExplainer(IPS_text_model['catboostclassifier'],
                feature_names=IPS_text_model['tfidfvectorizer'].get_feature_names_out())

pickle.dump(sql_explainer, open(os.path.join(explainer_save_path, 'DSS_IPS_shap_explainer.pkl'), 'wb'))
pickle.dump(text_explainer, open(os.path.join(explainer_save_path, 'DSS_IPS_text_shap_explainer.pkl'), 'wb'))
