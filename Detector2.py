# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Copyright (c) antillia.com Toshiyuki Arai
# 2020/08/12

# Detector2.py

import glob
import pathlib
import os.path
import traceback
from pathlib import Path
import sys
from detectron2.utils.logger import setup_logger


from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer

from detectron2.data import MetadataCatalog
import cv2
import torch

import detectron2.utils.visualizer as vis

import FiltersParser as fp

from FiltersParser import FiltersParser
from CocoInstanceSegmentation import *
from CocoPanopticSegmentation import *

from FilteredVisualizer import FilteredVisualizer


class Detector2:
  def __init__(self,  model, instance_mode=ColorMode,  metadata=None):  
    self.config    = model.config
    
    self.predictor = DefaultPredictor(self.config)
    self.instance_mode = instance_mode
 
    self.metadata      = metadata
    self.cpu_device    = torch.device("cpu")

    self.INSTANCES    = "instances"
    self.PANOPTIC_SEG = "panoptic_seg"
    self.SEM_SEG      = "sem_seg"



  # Detect each image in input_image_dir, and save detected image to output_dir
  def detect_all(self, input_image_dir, output_image_dir, filename_prefix, filters):
    image_list  = []

    if os.path.isdir(input_image_dir):
      image_list.extend(glob.glob(os.path.join(input_image_dir, "*.png")) )
      image_list.extend(glob.glob(os.path.join(input_image_dir, "*.jpg")) )

    print("image_list {}".format(image_list) )
    if not os.path.exists(output_image_dir):
      os.makedirs(output_image_dir)
        
    for image_filename in image_list:
      #image_filename will take images/foo.png
      image_file_path = os.path.abspath(image_filename)
      
      print("filename {}".format(image_file_path))
      
      self.detect(image_file_path, output_image_dir, filename_prefix, filters)


  def detect(self, image_filepath, output_image_dir, filename_prefix, filters=None):
    #OpenCV image in BGR format
    image = cv2.imread(image_filepath)

    predictions = self.predictor(image)

    #Convert image to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    catalog = MetadataCatalog.get(self.config.DATASETS.TRAIN[0])
   
    (vis_output, detected_objects) = self.visualize(filters, predictions, image, catalog, 1.2, self.instance_mode)

    image = vis_output.get_image()
 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if output_image_dir is not None:
      basename = os.path.basename(image_filepath)
      prefixed_filename = filename_prefix + basename
      output_filename = os.path.join(output_image_dir, prefixed_filename)

      cv2.imwrite(output_filename, image)
      print("Saved detected image to {}".format(output_filename))

      detected_objects_path = output_filename + '.txt'
      self.save_detected_objects(detected_objects, detected_objects_path)
      
    else:
      cv2.imshow('Results', image)


  def save_detected_objects(self, detected_objects, detected_objects_path):
    if detected_objects is None:
      return   
    print("Saved detected_objects to {}".format(detected_objects_path))

    # Save detected_objects data to a detected_objects_path file.
    # [(1, 'car:90%'), (2, 'person:80%'),... ]  
    with open(detected_objects_path, mode='w') as f:
      for item in detected_objects:
         (id, label_with_score) = item
         line = str(id) + " " + label_with_score +  "\n"
         f.write(line)
               
               
  def visualize(self, filters, predictions, image, metadata, scale, instance_mode):
    visualizer = FilteredVisualizer(filters, image, metadata=metadata, scale=scale, instance_mode=self.instance_mode)
    
    vis_output       = None
    detected_objects = None
    
    if self.PANOPTIC_SEG in predictions:
      panoptic_seg, segments_info = predictions[self.PANOPTIC_SEG]
      vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
    elif self.SEM_SEG  in predictions:
      vis_output = visualizer.draw_sem_seg(
                    predictions[self.SEM_SEG].argmax(dim=0).to(self.cpu_device)
                )
    elif self.INSTANCES in predictions:
      #Currently, INSTANCES case only support filters  
      instances = predictions[self.INSTANCES].to(self.cpu_device)
      (vis_output, detected_objects) = visualizer.draw_instance_predictions_with_filters(filters, predictions=instances)

    return (vis_output, detected_objects)



def parse_argv(argv):
    #The following img.png is taken from 
    # 'https://user-images.githubusercontent.com/11736571/77320690-099af300-6d37-11ea-9d86-24f14dc2d540.png'
    input_image_path = "./images/img.png"
    output_image_dir = None
    str_filters      = None
    filters          = None
    if len(argv) >= 2:
      input_image_path = argv[1]
        
    if len(argv) >= 3:
      output_image_dir = argv[2]

    if len(argv) == 4:
      # Specify a string like this [person,motorcycle] or "[person,motorcycle]" ,
      str_filters = argv[3]
      filtersParser = FiltersParser(str_filters, fp.COCO_CLASSES)
      filters = filtersParser.get_filters()
    print(filters)
                          
    if not os.path.exists(input_image_path):
        print("Not found {}".format(input_image_path))
        raise Exception("Not found {}".format(input_image_path))
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)                  
        
    return (input_image_path, output_image_dir, filters)
                   


if __name__ == "__main__":
  filename_prefix = "instance_seg_"
  try:
    (image_filepath, out_image_dir, filters) = parse_argv(sys.argv)
    
    model    = CocoInstanceSegmentation()
    #model    = CocoPanopticSegmentation()
    
    detector = Detector2(model)
    if os.path.isfile(image_filepath):
      detector.detect(image_filepath, out_image_dir, filters)
      
    elif os.path.isdir(input_image_filepath):
      detector.detect_all(image_filepath, out_image_dir, filters)

    else:
      raise Exception("Unsupported imput_image {}".format(input_image_filepath))

  except:
    traceback.print_exc()

    pass
     
