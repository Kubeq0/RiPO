"""
@author: JiXuan Xu, Jun Wang
@date: 20201023
@contact: jun21wangustc@gmail.com 
"""
import logging.config
logging.config.fileConfig("config/logging.conf")
logger = logging.getLogger('sdk')

import torch

# 🔧 TO DODAJ – potrzebne do załadowania modelu w PyTorch 2.6+
from models.network_def.mobilev3_pfld import PFLDInference
torch.serialization.add_safe_globals({'PFLDInference': PFLDInference})

from core.model_loader.BaseModelLoader import BaseModelLoader

class FaceAlignModelLoader(BaseModelLoader):
    def __init__(self, model_path, model_category, model_name, meta_file='model_meta.json'):
        logger.info('Start to analyze the face landmark model, model path: %s, model category: %s，model name: %s' %
                    (model_path, model_category, model_name))
        super().__init__(model_path, model_category, model_name, meta_file)
        self.cfg['img_size'] = self.meta_conf['input_width']

    def load_model(self):
        try:
            # 🔧 Ważne: weights_only=False aby załadować cały model (z klasą)
            model = torch.load(self.cfg['model_file_path'], weights_only=False)
        except Exception as e:
            logger.error('The model failed to load, please check the model path: %s!' %
                         self.cfg['model_file_path'])
            raise e
        else:
            logger.info('Successfully loaded the face landmark model!')
            return model, self.cfg
