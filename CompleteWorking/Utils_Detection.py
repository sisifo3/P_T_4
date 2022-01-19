coco_names = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

import cv2
import threading
class Camera:
    def __init__(self, rtsp_link=''):
        self.rtsp_link = rtsp_link
        self.capture = cv2.VideoCapture(self.rtsp_link,cv2.CAP_FFMPEG)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 0)
        self.capture.set( cv2.CAP_PROP_FPS, 10 )
        self.last_frame = None
        self.last_ready = False
        self.online = False
        self.load_network_stream()
        self.thread = threading.Timer(1./15., self.rtsp_cam_buffer)
        self.thread.setDaemon(True)

    def restart(self):
        self.online = False

    def load_network_stream(self):
        '''Verifies stream link and open new stream if valid'''
        print('load_network_stream')
        if self.verify_network_stream(self.rtsp_link):
            self.capture = cv2.VideoCapture(self.rtsp_link,cv2.CAP_FFMPEG)
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 0)
            self.capture.set( cv2.CAP_PROP_FPS, 15 )
            self.online = True

    def verify_network_stream(self, link):
        '''Attempts to receive a frame from given link'''
        cap = cv2.VideoCapture(link,cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
        cap.set( cv2.CAP_PROP_FPS, 15 )
        if not cap.isOpened():
            del cap
            return False
        cap.release()
        del cap
        return True
    def read(self):
        return self.last_ready, self.last_frame

    def rtsp_cam_buffer(self):
        while True:
            if self.capture.isOpened() and self.online:
                self.last_ready, self.last_frame = self.capture.read()
                if not self.last_ready:
                    self.capture.release()
                    self.online = False
                    #cv2.destroyAllWindows()
            else:
                # Attempt to reconnect
                print('attempting to reconnect', self.rtsp_link)
                self.load_network_stream()

###############################################################
#################################################
########################


import cv2
import sys
import os
import torch
from torch import from_numpy, no_grad
import numpy as np

import torchvision.transforms as transforms

#=====MASK RCNN

import cv2
import numpy as np
import random
# this will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

def draw_boxes(boxes, classes, labels, image):
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        cv2.putText(image, classes[i], (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                    lineType=cv2.LINE_AA)
    return image

import cv2
import torchvision
def draw_img_bbox(img,target):
    boxes = []
    for box in (target['boxes']):
        x, y, width, height  = box[0], box[1], box[2]-box[0], box[3]-box[1]
        img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0,255,0), 1)
        boxes.append(box)
    return img,boxes

def get_bbox(target):
    return target['boxes']
# the function takes the original prediction and the iou threshold.

def apply_nms(orig_prediction, iou_thresh=0.3):

    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)

    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]

    return final_prediction


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath("")
    #print(os.path.join(base_path, relative_path))
    return os.path.join(base_path, relative_path)


class MouseControl(object):
    def __init__(self):
        # self.points = []
        self.refPt = []
        self.retRect = []

    def click_and_crop(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            self.refPt.append((x, y))
            if len(self.refPt) > 2:
                self.refPt = []
                self.refPt.append((x, y))


import torchvision
# from .utils import *
# from .coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, maskrcnn_resnet50_fpn
from torch import cuda, backends
from datetime import datetime


# model_arc: which model to use, options are:
#            R-CNN: the model used is Faster R-CNN https://arxiv.org/abs/1506.01497
#            Mask-RCNN: the model used is Mask R-CNN https://arxiv.org/abs/1703.06870
# save_path: path where images will be saved, default folder is Documents/Dataset/,
#            don't forget to ad / at the end of the path.
class Detector(object):
    def __init__(self, model_arc='RCNN', save_path=os.path.expanduser('~') + '/Documents/Dataset/test/people'):
        super().__init__()
        self.save_path = save_path
        # check if path exists or make the directory if not.
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)
        # choose which model to use
        model_zoo = {'RCNN': fasterrcnn_resnet50_fpn,
                     'M-RCNN': maskrcnn_resnet50_fpn}
        # load weights from file instead from the internet
        # (useful when access is not granted)
        self.model = model_zoo[model_arc](
            pretrained=False, pretrained_backbone=False, num_classes=91)
        self.model.load_state_dict(torch.load(resource_path(
            '/home/sisifo/Desktop/{}_W.pth').format(model_arc)))

        # check CUDA availability
        self.CUDA = cuda.is_available()

        if self.CUDA:
            cuda.empty_cache()
            backends.cudnn.benchmark = True
            backends.cudnn.deterministic = True
            self.model.cuda()

        self.model.eval()

    # orig_img: cv2 image captured from stream/webcam
    # threshold: confidence threshold
    # filterPerson: (True) choose to filter bounding boxes of person
    #               (False) keep bounding boxes of all objects
    @torch.no_grad()
    def detect(self, orig_img, threshold=0.5, filterPerson=False):
        img2process = cv2.cvtColor(orig_img.copy(), cv2.COLOR_BGR2RGB)
        img2process = from_numpy(img2process.transpose(
            2, 0, 1)).float().div(255.0).unsqueeze(0)
        if self.CUDA:
            img2process = img2process.cuda()

        prediction = self.model(img2process)
        # get score for all the predicted objects
        scores = list(prediction[0]['scores'].detach().cpu().numpy())
        # index of those scores which are above a certain threshold
        thresholded_preds_inidices = [
            scores.index(i) for i in scores if i > threshold]
        thresholded_preds_count = len(thresholded_preds_inidices)

        # get all the predicted bounding boxes
        pred_bboxes = prediction[0]['boxes'].detach().cpu().numpy()
        # get boxes above the threshold score
        # boxes = pred_bboxes[pred_bboxes > detection_threshold]
        boxes = pred_bboxes[:thresholded_preds_count]
        labels = prediction[0]['labels'].detach().cpu()
        labels = labels[:thresholded_preds_count]

        # get all the predicited class names
        pred_classes = [coco_names[i] for i in prediction[0]['labels'].cpu().numpy()]
        pred_classes = pred_classes[:thresholded_preds_count]
        if filterPerson:
            pred_classes, boxes, labels = self.filter_outputs(pred_classes, boxes, labels)
        return pred_classes, boxes, labels

    # masks: when using Mask-RCNN model it stores the masks of the detected objects
    #        if the model used is RCNN, it returns prediction confidences and is not used.
    # boxes: (xmin,ymin,xmax,ymax) of the bounding box for each object detected.
    # labels: contains the label of the corresponding bounding box
    def filter_outputs(self, masks, boxes, labels, label2filter='person'):
        masksOr_classes = []
        boxes_ = []
        labels_ = []
        for i in range(len(masks)):
            if coco_names[labels[i].detach().cpu()] == label2filter:
                masksOr_classes.append(masks[i])
                boxes_.append(boxes[i])
                labels_.append(labels[i])

        return masksOr_classes, boxes_, labels_

    # crop person from Region Of Interest using its bounding box
    # img: image to crop.
    # boxes: bounding boxes of detected objects.
    # save: save image to file, used for building dataset. Defaults to True.
    def get_person_bbox(self, img, boxes, save=True, person_list=list()):
        timestampsave = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        for i, box in enumerate(boxes):
            # cropping with NumPy
            # startY:endY, startX:endX
            roi = img[int(box[1]):int(box[3]),
                  int(box[0]):int(box[2])]
            person_list.append(roi)
            if save:
                cv2.imwrite(self.save_path + timestampsave + '_bbox-{}.png'.format(i), roi)

            # print(self.save_path)

