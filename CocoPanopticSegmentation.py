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
# COCOInstanceSegmentation.py

import traceback

from detectron2.config import get_cfg
from DetectronModelZoo import DetectronModelZoo


class CocoPanopticSegmentation(DetectronModelZoo):

  def __init__(self, device="cpu", key="COCO-PanopticSegmentation/panoptic_fpn_R_50_3x" ):
    super().__init__() 
 
    self.segmentation   = key
    try:

      self.config   = get_cfg()
      (config_file, model_weights) = self.get(key)

      print(config_file)
      print(model_weights)
            
      self.config.merge_from_file(config_file) 
      self.config.merge_from_list(["MODEL.DEVICE", device, "MODEL.WEIGHTS", model_weights])
      self.config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
      self.config.freeze()

    except:
        traceback.print_exc()

