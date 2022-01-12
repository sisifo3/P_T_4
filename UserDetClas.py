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

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
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
# from utils.utils import *
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
import os
import glob
from Class_ClassificationModel_v3 import ClassificationModelResNet
from class_DetectionModel_v2 import PeopleDetector

class MainWindow(QMainWindow):

    '''
    if __name__ == "__main__":
        detect = PeopleDetector()
        detect.run()
    '''
    """docstring for Login."""

    def __init__(self):
        super().__init__()
        self.ui = loadUiWidget(resource_path("./ui_files/monitor_es_5.ui"))
        # Set starting image
        self.reconectando = np.zeros((600, 800, 3), np.uint8)
        self.reconectando[:, :, :] = 97
        sizeTxt = cv2.getTextSize('Reconectando...1', cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        x_, y_ = (self.reconectando.shape[1] - sizeTxt[0][0]) // 2, (self.reconectando.shape[0] - sizeTxt[0][1]) // 2
        cv2.putText(self.reconectando, 'Without frame', (x_, y_), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        ####### put frame : "without fame" ###############
        frame = self.reconectando
        image = QImage(frame, frame.shape[1], frame.shape[0],
                       frame.strides[0], QImage.Format_RGB888)
        self.ui.camera_feed.setPixmap(QPixmap.fromImage(image))
        ################ end ###########################
        self.width, self.height = 800, 600
        self.model_arc = 'RCNN'
        self.detector = Detector(model_arc=self.model_arc, save_path='Dataset/')

        self.capture_list = Camera('rtsp://visitante1:Ed$D-12220111@10.15.32.21/stream2')
        self.capture_list.thread.start()
        print("keep going")
        '''
        ######## run detector ##############
        
        detect = PeopleDetector()
        detect.run()
        ############# end ################
        '''

        image = QImage(frame, frame.shape[1], frame.shape[0],
                       frame.strides[0], QImage.Format_RGB888)
        self.ui.camera_feed.setPixmap(QPixmap.fromImage(image))

        print("keep going 2")
        self.thread = threading.Thread(target=self.run_d, name="FasterRCNN_thread")
        self.thread.daemon = True
        self.thread.start()
        print("keep going for ever")

        '''
        ######## run Classficator ##############
        a , b = ClassificationModelResNet()
        print("a : ", a)
        print("b :", b)
        ############# end ################

        self.ui.maximoalmomento.setText('{}'.format(a))
        self.ui.hora_pico.setText('{}'.format(b))
        '''

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

                    # sitting_people = 0
                    # standing_people = 1
                    print("sitting_people: ", c_sit_peo)
                    print("standing_people: ", c_stt_peo)
                    totalPeopleDetected = c_sit_peo + c_stt_peo
                    self.ui.maximoalmomento.setText('{}'.format(c_sit_peo))
                    self.ui.hora_pico.setText('{}'.format(c_stt_peo))
                    self.ui.ppl_count.setText('{}'.format(totalPeopleDetected))

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


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    widget = QStackedWidget()
    widget.setMinimumSize(480, 290)  # 480, 170)
    widget.setWindowTitle('People Detection and Classification')
    #widget.setWindowIcon(QIcon(resource_path('./ui_files/logoicon.ico')))
    mainwindow_ = MainWindow()
    widget.addWidget(mainwindow_.ui)
    widget.show()


    sys.exit(app.exec_())
    # app.exec_()

