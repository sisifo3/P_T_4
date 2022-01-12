
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


class PeopleDetector(object):
    def __init__(self):
        super().__init__()
        self.width, self.height = 800, 600
        self.model_arc = 'RCNN'
        self.detector = Detector(model_arc=self.model_arc, save_path='Dataset/')

        self.capture_list = Camera(
            # 'rtsp://visitante1:Ed$D-12220111@10.15.32.18/stream2')  # change username, password and IP address
            'rtsp://visitante1:Ed$D-12220111@10.15.32.21/stream2')  # change username, password and IP address
        # 'rtsp://visitante1:Ed$D-12220111@10.15.32.32/stream2')  # change username, password and IP address
        # 'rtsp://visitante1:Ed$D-12220111@10.15.32.33/stream2')  # change username, password and IP address
        self.capture_list.thread.start()

    def run(self):
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
                    #list_cropped_people = self.detector.get_person_bbox_list(img=frame, boxes=boxes)

                    for i, (val) in enumerate(person_list):
                        path_t = '/home/sisifo/PycharmProjects/ML/venv/Utils_3/dataset_C/test/people/'
                        cv2.imwrite(path_t + 'test_img_' + str(i) + '.png', val)

                    ClassificationModelResNet()

                    # sitting_people = 0
                    # standing_people = 1
                    #print("sitting_people: ", a)
                    #print("standing_people: ", b)

                    ########## Delete temporal files#########
                    files = glob.glob('/home/sisifo/PycharmProjects/ML/venv/Utils_3/dataset_C/test/people/*')
                    for f in files:
                        os.remove(f)
                    ######### end deleted###########

                    # draw detections
                    for box in boxes:
                        frame = cv2.rectangle(frame, (int(box[0]), int(
                            box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                cv2.imshow('frame', frame)
                #return _, frame
                # change 1 to X ms if you want to let it wait for X ms before
                # processing one more frame.
                k = cv2.waitKey(1)
                if k == 27:
                    break


#if __name__ == "__main__":
#    detect = PeopleDetector()
#    detect.run()

