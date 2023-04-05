import shap
from setting import *
import torch
import scipy as sp
import numpy as np
import os

explainer_save_path = 'YOUR SHAP EXPLAINER SAVE DIR !!!!!!!'

ips_explainer = shap.TreeExplainer(IPS_model)
waf_explainer = shap.TreeExplainer(WAF_model)
web_explainer = shap.TreeExplainer(WEB_model)


pickle.dump(ips_explainer, open(os.path.join(explainer_save_path, 'DSS_IPS_LGB_explainer_20230313.pkl'), 'wb'))
pickle.dump(waf_explainer, open(os.path.join(explainer_save_path, 'DSS_WAF_LGB_explainer_20230313.pkl'), 'wb'))
pickle.dump(web_explainer, open(os.path.join(explainer_save_path, 'DSS_WEB_LGB_explainer_20230404.pkl'), 'wb'))

