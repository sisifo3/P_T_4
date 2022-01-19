
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
from Utils_dataStock.Utils_data import read_stock_data, basic_search

class MainWindow(QMainWindow):
    from Utils_Model_Det_Class.ModelsRun import run_d
    from Utils_dataStock.Utils_data import basic_search, plot_sitting_people,plot_sitting_people_month, \
        Analitic_day, Analitic_period_time

    """docstring for Login."""

    def __init__(self):
        super().__init__()
        self.ui = loadUiWidget(resource_path("./ui_files/monitor_es_6.ui"))

        img = cv2.imread(resource_path("./ui_files/ITESM-Logo.png"))  # , cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (281, 74), cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img[img == (255, 255, 255)] = (240)
        logo = QImage(img,
                      img.shape[1],
                      img.shape[0],
                      img.strides[0],
                      QImage.Format_RGB888)
        self.ui.logo.setPixmap(QPixmap.fromImage(logo))

        img = cv2.imread(resource_path("./ui_files/img_2.png"))  # , cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (570, 203), cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img[img == (255, 255, 255)] = (240)
        logo = QImage(img,
                      img.shape[1],
                      img.shape[0],
                      img.strides[0],
                      QImage.Format_RGB888)
        self.ui.logo_2.setPixmap(QPixmap.fromImage(logo))

        img = cv2.imread(resource_path("./ui_files/img_3.png"))  # , cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (570, 203), cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img[img == (255, 255, 255)] = (240)
        logo = QImage(img,
                      img.shape[1],
                      img.shape[0],
                      img.strides[0],
                      QImage.Format_RGB888)
        self.ui.logo_3.setPixmap(QPixmap.fromImage(logo))

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
        # '''
        ######## run detector ##############
        self.width, self.height = 800, 600
        self.model_arc = 'RCNN'
        self.detector = Detector(model_arc=self.model_arc, save_path='Dataset/')
        self.capture_list = Camera('rtsp://visitante1:Ed$D-12220111@10.15.32.21/stream2')
        self.capture_list.thread.start()
        print("keep going")
        print("keep going 2")
        self.thread = threading.Thread(target=self.run_d, name="FasterRCNN_thread")
        self.thread.daemon = True
        self.thread.start()
        print("keep going for ever")
        ############# end ################
        # '''

        '''
        ######## get day and time ########## 
        date_temp = self.ui.dateEdit_2.text()
        time_temp = self.ui.timeEdit.text()
        print(type(date_temp))
        print(type(time_temp))
        ######### end get day and time########
        '''
        a = 'Hola!'
        # print(a)
        self.ui.label_12.setText('{}'.format(a))
        date_temp = self.ui.lineEdit.text()
        time_temp = self.ui.lineEdit_2.text()
        #date_temp = "01/05/2020"
        #time_temp = "15:00:00"

        self.ui.pushButton.clicked.connect(self.basic_search)
        self.ui.pushButton_5.clicked.connect(self.plot_sitting_people)
        self.ui.pushButton_6.clicked.connect(self.plot_sitting_people_month)
        self.ui.pushButton_7.clicked.connect(self.Analitic_day)
        self.ui.pushButton_9.clicked.connect(self.Analitic_period_time)








if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    widget = QStackedWidget()
    widget.setMinimumSize(790, 590)  # 480, 170)
    widget.setWindowTitle('Monitoreo de Espacios Colaborativos y Académicos Inteligentes')
    widget.setWindowIcon(QIcon(resource_path('./ui_files/logoicon.ico')))
    mainwindow_ = MainWindow()
    widget.addWidget(mainwindow_.ui)
    widget.show()
    sys.exit(app.exec_())
    # app.exec_()

