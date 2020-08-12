# 
# DetectronModelZoo.py
# Copyright (c) antillia.com Toshiyuki Arai

import traceback

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.model_zoo  import *
from detectron2.model_zoo.model_zoo  import _ModelZooUrls
import detectron2.model_zoo.model_zoo  as mz


class DetectronModelZoo:

  def __init__(self):
    self.DETECTRON2 = "detectron2://"

  def get(self, key):
    config_file = ""
    weight_file = ""

    for i, config_path in enumerate(_ModelZooUrls.CONFIG_PATH_TO_URL_SUFFIX):
      #print(config_path)
      if key in config_path:
        config_file = mz.get_config_file(config_path)
        weight = _ModelZooUrls.CONFIG_PATH_TO_URL_SUFFIX[config_path]
        array = config_path.split('.')
        if len(array) >0:
          weight_file = self.DETECTRON2+ array[0] + "/" + weight
          #Maybe the following is better. 
          #weight_file = mz.get_checkpoint_url(config_path)
        break

    return (config_file, weight_file)
    
