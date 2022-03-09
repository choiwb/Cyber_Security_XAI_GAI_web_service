import os
import pickle
import shap
from setting import *


explainer_save_path = 'SHAP explainer save path !!!'

explainer = shap.TreeExplainer(IPS_model)

pickle.dump(explainer, open(os.path.join(explainer_save_path, 'DSS_IPS_shap_explainer.pkl'), 'wb'))
