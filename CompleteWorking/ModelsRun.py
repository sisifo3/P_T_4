from utils.ui_utils import loadUiWidget, EventMouse, EventMouse_click, Settings, CalibrationFile, userFile, \
    resource_path, EventHandler

from PySide2.QtWidgets import QDialog
import sys
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtGui import QImage
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

'''cv2 isn'n compatible with PyQt5, for this reason use os.envoriment'''
import cv2
import os

'''end'''

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.colors import ColorConverter
from matplotlib.legend import _get_legend_handles_labels
from torch import cuda, backends
from torchvision.models import detection
import torch
import numpy as np
from numpy import ceil
import threading
from threading import Lock
from datetime import datetime

#### from Detection and classifcation#######
import __future__
__future__.division
import cv2
import os
from Utils_Model_Det_Class.Utils_Detection import *

import os
import glob
from Class_ClassificationModel_v3 import ClassificationModelResNet
from class_DetectionModel_v2 import PeopleDetector
from Utils_dataStock.Utils_data import read_stock_data


def run_d(self):
    while True:
        _, orig_img = self.capture_list.read()
        if _ and orig_img is not None:
            # Resize
            orig_img = cv2.resize(orig_img, (self.width, self.height))
            # copy original image as it is going to be modified and loaded to GPU if available
            frame = orig_img.copy()

            # Detect person in image
            pred_classes, boxes, labels = self.detector.detect(
                orig_img, filterPerson=True)
            if len(boxes) > 0:
                # person_lists: holds roi of people detected
                person_list = list()
                self.detector.get_person_bbox(
                    img=frame, boxes=boxes, save=True, person_list=person_list)

                # here you can add your classification model and pass person_list to it
                # after been moved to GPU. get_person_bbox_list

                for i, (val) in enumerate(person_list):
                    path_t = '/home/sisifo/PycharmProjects/ML/venv/Utils_3/dataset_C/test/people/'
                    cv2.imwrite(path_t + 'test_img_' + str(i) + '.png', val)

                c_sit_peo, c_stt_peo = ClassificationModelResNet()
                zone = 1
                read_stock_data(zone, c_sit_peo, c_stt_peo)

                # sitting_people = 0
                # standing_people = 1
                print("sitting_people: ", c_sit_peo)
                print("standing_people: ", c_stt_peo)
                totalPeopleDetected = c_sit_peo + c_stt_peo
                As_disp = 50 - totalPeopleDetected
                self.ui.maximoalmomento.setText('{}'.format(c_sit_peo))
                self.ui.hora_pico.setText('{}'.format(c_stt_peo))
                self.ui.ppl_count.setText('{}'.format(totalPeopleDetected))
                self.ui.hora_pico_3.setText('{}'.format(As_disp))

                ########## Delete temporal files#########
                files = glob.glob('/home/sisifo/PycharmProjects/ML/venv/Utils_3/dataset_C/test/people/*')
                for f in files:
                    os.remove(f)
                ######### end deleted###########

                # draw detections
                for box in boxes:
                    frame = cv2.rectangle(frame, (int(box[0]), int(
                        box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

            image = QImage(frame, frame.shape[1], frame.shape[0],
                           frame.strides[0], QImage.Format_RGB888)
            self.ui.camera_feed.setPixmap(QPixmap.fromImage(image))
