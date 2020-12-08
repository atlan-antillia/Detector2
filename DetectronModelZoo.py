#******************************************************************************
#
#  Copyright (c) 2020-2021 Antillia.com TOSHIYUKI ARAI. ALL RIGHTS RESERVED.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#******************************************************************************

# 
# DetectronModelZoo.py

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
    
