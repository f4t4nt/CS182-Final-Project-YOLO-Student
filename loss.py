"""
Implementation of Yolo Loss Function from the original yolo paper
Adapted from https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection/YOLO

"""

import torch
import torch.nn as nn
from utils import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=2, lambda_noobj=0.5, lambda_coord=5):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

        """
        S is split size of image (in paper 7),
        B is number of boxes (in paper 2),
        C is number of classes (in paper and VOC dataset is 20),
        """
        self.S = S
        self.B = B
        self.C = C

        # These are from Yolo paper, signifying how much we should
        # pay loss for no object (noobj) and the box coordinates (coord)
        self.lambda_noobj = lambda_noobj
        self.lambda_coord = lambda_coord

    def forward(self, predictions, target):
        # predictions are shaped (BATCH_SIZE, S*S(C+B*5) when inputted
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        # Calculate IoU for the two predicted bounding boxes with target bbox
        # hard coded for two boxes here
        # iou_b1 is shape [B,S,S,1]
        
        iou_b1 = intersection_over_union(predictions[..., (self.C+1):(self.C+5)], target[..., (self.C+1):(self.C+5)], box_format="yolo")
        iou_b2 = intersection_over_union(predictions[..., (self.C+6):(self.C+10)], target[..., (self.C+1):(self.C+5)], box_format="yolo")
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0) 
        # ious is shape [1,B,S,S,1] 
        # print(iou_b1[0])
        # Take the box with highest IoU out of the two prediction
        # Note that bestbox will be indices of 0, 1 for which bbox was best
        iou_maxes, bestbox = torch.max(ious, dim=0) # shape (B,S,S,1)
        exists_box = target[..., self.C].unsqueeze(3)  # in paper this is Iobj_i 

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        # Set boxes with no object in them to 0. We only take out one of the two 
        # predictions, which is the one with highest Iou calculated previously.
        box_predictions = exists_box * (
            (
                bestbox * predictions[..., (self.C+6):(self.C+10)]
                + (1 - bestbox) * predictions[..., (self.C+1):(self.C+5)]
            )
        )

        box_targets = exists_box * target[..., (self.C+1):(self.C+5)]

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )
        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        # pred_box is the confidence score for the bbox with highest IoU
        pred_box = (
            bestbox * predictions[..., (self.C+5):(self.C+6)] + (1 - bestbox) * predictions[..., self.C:(self.C+1)]
        )

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., self.C:(self.C+1)]),
        )
        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., self.C:(self.C+1)], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., self.C:(self.C+1)], start_dim=1),
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., (self.C+5):(self.C+6)], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., self.C:(self.C+1)], start_dim=1)
        )

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :self.C], end_dim=-2,),
            torch.flatten(exists_box * target[..., :self.C], end_dim=-2,),
        )

        ### YOUR CODE HERE
        # HINT: the loss should be a linear combination of 
        # MSE losses calculated above.
        loss = NotImplementedError()
        ### END CODE HERE

        return loss