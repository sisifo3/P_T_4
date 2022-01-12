from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *
from PySide2.QtUiTools import QUiLoader
import numpy as np
import os
import sys
def loadUiWidget(uifilename, parent=None):
    loader = QUiLoader()
    uifile = QFile(uifilename)
    uifile.open(QFile.ReadOnly)
    ui = loader.load(uifile, parent)
    uifile.close()
    return ui

class EventMouse(QObject):
    """docstring for EventMouse."""
    def __init__(self):
        super().__init__()
        self.pos = [[],[],[],[],[]]
    def eventFilter(self, obj, event):
        if event.type() == QEvent.MouseButtonPress:
            #print("Ate key press", event.pos())
            self.pos = (event.pos().x(),event.pos().y())
            if len(self.pos) > 4:
                self.pos = []
                self.pos.append(( event.pos().x(),event.pos().y() ))
            return (event.pos().x(),event.pos().y())#True
        #elif event.type() == QEvent.MouseButtonRelease:
        #    self.pos = []
class EventMouse_click(QObject):
    """docstring for EventMouse."""
    def __init__(self):
        super().__init__()
        self.pos = [[],[],[],[],[]]
        self.camSelected = 0
    def eventFilter(self, obj, event):
        if event.type() == QEvent.MouseButtonPress:
            #print("Ate key press", event.pos())
            self.pos[self.camSelected].append( (event.pos().x(),event.pos().y()) )
            if len(self.pos[self.camSelected]) > 4:
                self.pos[self.camSelected] = []
                self.pos[self.camSelected].append(( event.pos().x(),event.pos().y() ))
            return True


class EventHandler(QObject):
    """docstring for EventMouse."""

    def __init__(self):
        super().__init__()

    def eventFilter(self, obj, event):
        #print(event)
        if event.type() == QEvent.KeyPress:
            key = event.key()
            if key == Qt.Key_Escape:
                #print('escape')
                return True
        else:
            return super(EventHandler, self).eventFilter(obj, event)


class Settings(object):
    """docstring for Settings."""
    def __init__(self):
        super(Settings, self).__init__()
        self.camIP = ''
        self.camPass = '210823'
        self.camUser = 'V2021'
        self.camZone = ''
        self.camFolder = ''
        self.threshold = '70'
    def print(self):
        print(self.camIP)
        print(self.camPass)
        print(self.camUser)
        print(self.camZone)
        print(self.camFolder)
        print(self.threshold)

class CalibrationFile(object):
    """docstring for CalibrationFile."""
    def __init__(self):
        super(CalibrationFile, self).__init__()
        self.d1 = '0'
        self.d2 = '0'
        self.d3 = '0'
        self.d4 = '0'
        self.corner_points = np.zeros((4,2),dtype=np.float32)
    def print(self):
        print(self.d1)
        print(self.d2)
        print(self.d3)
        print(self.d4)
        print(self.corner_points)

class userFile(object):
    """docstring for userFile."""
    def __init__(self):
        super(userFile, self).__init__()
        self.username = ''
        self.password = ''
        self.name = ''
        self.usertype = 'Administrador'
    def setPass(self,password):
        return password+'passed'
        
def get_directory_size(directory):
    """Returns the `directory` size in bytes."""
    total = 0
    try:
        # print("[+] Getting the size of", directory)
        for entry in os.scandir(directory):
            if entry.is_file():
                # if it's a file, use stat() function
                total += entry.stat().st_size
            elif entry.is_dir():
                # if it's a directory, recursively call this function
                total += get_directory_size(entry.path)
    except NotADirectoryError:
        # if `directory` isn't a directory, get the file size then
        return os.path.getsize(directory)
    except PermissionError:
        # if for whatever reason we can't open the folder, return 0
        return 0
    return total

def get_size_format(b, factor=1024, suffix="B"):
    """
    Scale bytes to its proper byte format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if b < factor:
            return "{:.2f}{}{}".format(b,unit,suffix)
        b /= factor
    return "{:.2f}{}".format(b,suffix)


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    #print(os.path.join(base_path, relative_path))
    return os.path.join(base_path, relative_path)

