import torch
from torchmetrics import JaccardIndex
import torchvision
from torchvision import ops 
from torch import nn
from scipy.optimize import linear_sum_assigmeent
import cv2
import numpy as np 

class HungarianAlgorithm(nn.Module):

    super().__init__()
    self.bounding_box_cost = bounding_box_cost
    self.GIOU_cost = GIOU_cost
    self.classes_cost = classes_cost

    assert classes_cost != 0 or Generalized_IOU !=0 or bounding_box_cost !=0


    def forward(self, predicted, targets):

        
        "#TO-DO: create target bbox and target indices for labels
         #- compute cost for predicted bbox, giou and classes vs target

         target_bounding_box = torch.cat([ground_truth["boxes"]])
         target_labels = ground_truth["labels"]

         bounding_box_cost = torch.cdist(bounding_boxes, target_bounding_box)
         classes_cost = -class_probs[:,target_labels]
         GIOU_cost = torch.(predicted "predicted and target bounding boxes expected to be in shape Tensor[N,4] i.e: (x1,x2,y1,y2) format" , target)


         Cost_matrix = self.bounding_box_cost * bounding_box_cost + self.GIOU_cost * GIOU_cost + self.classes_cost * classes_cost
         Cost = scipy.optimize.linear_sum_assignment(Cost_matrix, maximize=False)

path = "/Users/neilsuji/Downloads/screenshot_images/2022-06-01.png"



