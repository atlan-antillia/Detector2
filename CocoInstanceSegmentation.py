# 
# COCOInstanceSegmentation.py
# 2020/08/12 atlan
# Copyright (c) antillia.com Toshiyuki Arai

import traceback

from detectron2.config import get_cfg

from DetectronModelZoo import *


class CocoInstanceSegmentation(DetectronModelZoo):

  def __init__(self, device="cpu", key="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x" ):
    super().__init__() 
    self.segmentation   = key
    try:

      self.config   = get_cfg()
      #model = DetectronModelZoo()
      (config_file, model_weights) = self.get(key)

      print(config_file)
      print(model_weights)
            
      self.config.merge_from_file(config_file) 
      self.config.merge_from_list(["MODEL.DEVICE", device, "MODEL.WEIGHTS", model_weights])
      self.config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
      self.config.freeze()

    except:
        traceback.print_exc()

