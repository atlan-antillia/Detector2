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

# PanopticSegmentation.py

import os
import traceback
import sys

from CocoPanopticSegmentation import *
from Detector2 import Detector2
import Detector2 as det


if __name__ == "__main__":
  filename_prefix = "panoptic_"
  
  try:
    (image_filepath, out_image_dir, filters) = det.parse_argv(sys.argv)
    
    model    = CocoPanopticSegmentation()
    
    detector = Detector2(model)
    if os.path.isfile(image_filepath):
      detector.detect(image_filepath, out_image_dir, filename_prefix, filters)
      
    elif os.path.isdir(image_filepath):
      detector.detect_all(image_filepath, out_image_dir, filename_prefix, filters)

    else:
      raise Exception("Unsupported imput_image {}".format(input_image_filepath))

  except:
    traceback.print_exc()

     
