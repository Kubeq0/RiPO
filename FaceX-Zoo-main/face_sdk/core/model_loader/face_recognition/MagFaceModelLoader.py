"""
@author: JiXuan Xu, Jun Wang
@date: 20201015
@contact: jun21wangustc@gmail.com
"""
import sys
import os
import hashlib

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, BASE_DIR)
import logging.config
logging.config.fileConfig("config/logging.conf")
logger = logging.getLogger('sdk')

import torch

# 🔧 Dodaj DataParallel do bezpiecznych globali
from torch.nn.parallel.data_parallel import DataParallel
torch.serialization.add_safe_globals({'DataParallel': DataParallel})

from core.model_loader.BaseModelLoader import BaseModelLoader

class MagFaceModelLoader(BaseModelLoader):
    def __init__(self, model_path, model_category, model_name, meta_file='model_meta.json'):
        logger.info('Start to analyze the face recognition model, model path: %s, model category: %s，model name: %s' %
                    (model_path, model_category, model_name))
        super().__init__(model_path, model_category, model_name, meta_file)
        self.cfg['mean'] = self.meta_conf['mean']
        self.cfg['std'] = self.meta_conf['std']

    def load_model(self):
        try:
            # Załaduj checkpoint
            checkpoint = torch.load(self.cfg['model_file_path'], map_location='cpu')

            # Pobierz state_dict – obsłuż sytuację, gdy checkpoint ma dodatkowe klucze
            state_dict = checkpoint.get('state_dict', checkpoint)

            # Usuń ewentualny prefiks (np. 'backbone.' lub 'module.') z kluczy
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("backbone."):
                    k = k[len("backbone."):]
                elif k.startswith("module."):
                    k = k[len("module."):]
                new_state_dict[k] = v

            # 🔧 Załaduj odpowiednią architekturę zgodną z treningiem (np. iresnet50)
            from backbone.ResNets import Resnet
            model = Resnet(num_layers=152, drop_ratio=0.4, mode="ir_se")  # lub 1000/None zależnie od Twojego treningu

            model.load_state_dict(new_state_dict, strict=False)

            model = model.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
            model.eval()

        except Exception as e:
            logger.error('The model failed to load: %s' % str(e))
            raise e
        else:
            logger.info('Successfully loaded the face recognition model!')
            return model, self.cfg

