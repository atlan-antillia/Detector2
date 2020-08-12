#
# Copyright (c) antillia.com Toshiyuki Arai
# 2020/08/12

# InstanceSegmentation.py

import os
import traceback
import sys

from CocoInstanceSegmentation import *
from Detector2 import Detector2

import Detector2 as det


if __name__ == "__main__":
  filename_prefix = "instance_seg_"
  
  try:
    (image_filepath, out_image_dir, filters) = det.parse_argv(sys.argv)
    
    model    = CocoInstanceSegmentation()
    
    detector = Detector2(model)
    if os.path.isfile(image_filepath):
      detector.detect(image_filepath, out_image_dir, filename_prefix, filters)
      
    elif os.path.isdir(image_filepath):
      detector.detect_all(image_filepath, out_image_dir, filename_prefix, filters)

    else:
      raise Exception("Unsupported imput_image {}".format(input_image_filepath))

  except:
    traceback.print_exc()

    pass
     
