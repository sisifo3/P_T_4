import sys
import os
import time
import math
import torch
from torch import min as torchmin 
from torch import max as torchmax 
from torch import from_numpy, no_grad,zeros,sort,FloatTensor,linspace,sigmoid,Tensor,LongTensor,exp,nn,ByteStorage,ByteTensor
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable
from utils.coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names
import torchvision.transforms as transforms
import itertools
import struct  # get_image_size
import imghdr  # get_image_size
import cv2


def Sigmoid(x):
    return 1.0 / (np.exp(-x) + 1.)


def softmax(x):
    x = np.exp(x - np.expand_dims(np.max(x, axis=1), axis=1))
    x = x / np.expand_dims(x.sum(axis=1), axis=1)
    return x


def bbox_iou(box1, box2, x1y1x2y2=True):
    #h = 480#352#480
    #w = 640#480#640
    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        mx = min(box1[0] - box1[2] / 2.0, box2[0] - box2[2] / 2.0)
        Mx = max(box1[0] + box1[2] / 2.0, box2[0] + box2[2] / 2.0)
        my = min(box1[1] - box1[3] / 2.0, box2[1] - box2[3] / 2.0)
        My = max(box1[1] + box1[3] / 2.0, box2[1] + box2[3] / 2.0)
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    
    if max(w1,w2) > 0.23:# or max(h1,h2)>0.3:
        return 1.

    return carea / uarea


def bbox_ious(boxes1, boxes2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = torchmin(boxes1[0], boxes2[0])
        Mx = torchmax(boxes1[2], boxes2[2])
        my = torchmin(boxes1[1], boxes2[1])
        My = torchmax(boxes1[3], boxes2[3])
        w1 = boxes1[2] - boxes1[0]
        h1 = boxes1[3] - boxes1[1]
        w2 = boxes2[2] - boxes2[0]
        h2 = boxes2[3] - boxes2[1]
    else:
        mx = min(boxes1[0] - boxes1[2] / 2.0, boxes2[0] - boxes2[2] / 2.0)
        Mx = max(boxes1[0] + boxes1[2] / 2.0, boxes2[0] + boxes2[2] / 2.0)
        my = min(boxes1[1] - boxes1[3] / 2.0, boxes2[1] - boxes2[3] / 2.0)
        My = max(boxes1[1] + boxes1[3] / 2.0, boxes2[1] + boxes2[3] / 2.0)
        w1 = boxes1[2]
        h1 = boxes1[3]
        w2 = boxes2[2]
        h2 = boxes2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    mask = ((cw <= 0) + (ch <= 0) > 0)
    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    carea[mask] = 0
    uarea = area1 + area2 - carea
    return carea / uarea


def nms(boxes, nms_thresh):
    if len(boxes) == 0:
        return boxes

    det_confs = zeros(len(boxes))
    for i in range(len(boxes)):
        #det_confs[i] = 1 - boxes[i][4]
        det_confs[i] = 1 - boxes[i][4]

    _, sortIds = sort(det_confs,descending=False)
    out_boxes = []
    for i in range(len(boxes)):
        box_i = boxes[sortIds[i]]
        if box_i[4] > 0.6:
            out_boxes.append(box_i)
            for j in range(i + 1, len(boxes)):
                box_j = boxes[sortIds[j]]
                #b_iou = bbox_iou(box_i, box_j, x1y1x2y2=False)
                if bbox_iou(box_i, box_j, x1y1x2y2=False)  > nms_thresh:
                    # print(box_i, box_j, bbox_iou(box_i, box_j, x1y1x2y2=False))
                    box_j[4] = 0
    return out_boxes


def convert2cpu(gpu_matrix):
    return FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)


def convert2cpu_long(gpu_matrix):
    return LongTensor(gpu_matrix.size()).copy_(gpu_matrix)


def get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors, only_objectness=1, validation=False):
    anchor_step = len(anchors) // num_anchors
    if output.dim() == 3:
        output = output.unsqueeze(0)
    batch = output.size(0)
    assert (output.size(1) == (5 + num_classes) * num_anchors)
    h = output.size(2)
    w = output.size(3)

    t0 = time.time()
    all_boxes = []
    output = output.view(batch * num_anchors, 5 + num_classes, h * w).transpose(0, 1).contiguous().view(5 + num_classes,
                                                                                                        batch * num_anchors * h * w)

    grid_x = linspace(0, w - 1, w).repeat(h, 1).repeat(batch * num_anchors, 1, 1).view(
        batch * num_anchors * h * w).type_as(output)  # cuda()
    grid_y = linspace(0, h - 1, h).repeat(w, 1).t().repeat(batch * num_anchors, 1, 1).view(
        batch * num_anchors * h * w).type_as(output)  # cuda()
    xs = sigmoid(output[0]) + grid_x
    ys = sigmoid(output[1]) + grid_y

    anchor_w = Tensor(anchors).view(num_anchors, anchor_step).index_select(1, LongTensor([0]))
    anchor_h = Tensor(anchors).view(num_anchors, anchor_step).index_select(1, LongTensor([1]))
    anchor_w = anchor_w.repeat(batch, 1).repeat(1, 1, h * w).view(batch * num_anchors * h * w).type_as(output)  # cuda()
    anchor_h = anchor_h.repeat(batch, 1).repeat(1, 1, h * w).view(batch * num_anchors * h * w).type_as(output)  # cuda()
    ws = exp(output[2]) * anchor_w
    hs = exp(output[3]) * anchor_h

    det_confs = sigmoid(output[4])

    cls_confs = nn.Softmax()(Variable(output[5:5 + num_classes].transpose(0, 1))).data
    cls_max_confs, cls_max_ids = torchmax(cls_confs, 1)
    cls_max_confs = cls_max_confs.view(-1)
    cls_max_ids = cls_max_ids.view(-1)
    t1 = time.time()

    sz_hw = h * w
    sz_hwa = sz_hw * num_anchors
    det_confs = convert2cpu(det_confs)
    cls_max_confs = convert2cpu(cls_max_confs)
    cls_max_ids = convert2cpu_long(cls_max_ids)
    xs = convert2cpu(xs)
    ys = convert2cpu(ys)
    ws = convert2cpu(ws)
    hs = convert2cpu(hs)
    if validation:
        cls_confs = convert2cpu(cls_confs.view(-1, num_classes))
    t2 = time.time()
    for b in range(batch):
        boxes = []
        for cy in range(h):
            for cx in range(w):
                for i in range(num_anchors):
                    ind = b * sz_hwa + i * sz_hw + cy * w + cx
                    det_conf = det_confs[ind]
                    if only_objectness:
                        conf = det_confs[ind]
                    else:
                        conf = det_confs[ind] * cls_max_confs[ind]

                    if conf > conf_thresh:
                        bcx = xs[ind]
                        bcy = ys[ind]
                        bw = ws[ind]
                        bh = hs[ind]
                        cls_max_conf = cls_max_confs[ind]
                        cls_max_id = cls_max_ids[ind]
                        box = [bcx / w, bcy / h, bw / w, bh / h, det_conf, cls_max_conf, cls_max_id]
                        if (not only_objectness) and validation:
                            for c in range(num_classes):
                                tmp_conf = cls_confs[ind][c]
                                if c != cls_max_id and det_confs[ind] * tmp_conf > conf_thresh:
                                    box.append(tmp_conf)
                                    box.append(c)
                        boxes.append(box)
        all_boxes.append(boxes)
    t3 = time.time()
    if False:
        print('---------------------------------')
        print('matrix computation : %f' % (t1 - t0))
        print('        gpu to cpu : %f' % (t2 - t1))
        print('      boxes filter : %f' % (t3 - t2))
        print('---------------------------------')
    return all_boxes


def get_region_boxes1(output, conf_thresh, num_classes, anchors, num_anchors, only_objectness=1, validation=False):
    anchor_step = len(anchors) // num_anchors
    if len(output.shape) == 3:
        output = np.expand_dims(output, axis=0)
    batch = output.shape[0]
    assert (output.shape[1] == (5 + num_classes) * num_anchors)
    h = output.shape[2]
    w = output.shape[3]

    t0 = time.time()
    all_boxes = []
    output = output.reshape(batch * num_anchors, 5 + num_classes, h * w).transpose((1, 0, 2)).reshape(
        5 + num_classes,
        batch * num_anchors * h * w)

    grid_x = np.expand_dims(np.expand_dims(np.linspace(0, w - 1, w), axis=0).repeat(h, 0), axis=0).repeat(
        batch * num_anchors, axis=0).reshape(
        batch * num_anchors * h * w)
    grid_y = np.expand_dims(np.expand_dims(np.linspace(0, h - 1, h), axis=0).repeat(w, 0).T, axis=0).repeat(
        batch * num_anchors, axis=0).reshape(
        batch * num_anchors * h * w)

    xs = Sigmoid(output[0]) + grid_x
    ys = Sigmoid(output[1]) + grid_y

    anchor_w = np.array(anchors).reshape((num_anchors, anchor_step))[:, 0]
    anchor_h = np.array(anchors).reshape((num_anchors, anchor_step))[:, 1]
    anchor_w = np.expand_dims(np.expand_dims(anchor_w, axis=1).repeat(batch, 1), axis=2).repeat(h * w, axis=1).reshape(
        batch * num_anchors * h * w)  # cuda()
    anchor_h = np.expand_dims(np.expand_dims(anchor_h, axis=1).repeat(batch, 1), axis=2).repeat(h * w, axis=1).reshape(
        batch * num_anchors * h * w)  # cuda()
    ws = np.exp(output[2]) * anchor_w
    hs = np.exp(output[3]) * anchor_h

    det_confs = Sigmoid(output[4])

    cls_confs = softmax(output[5:5 + num_classes].transpose(1, 0))
    cls_max_confs = np.max(cls_confs, 1)
    cls_max_ids = np.argmax(cls_confs, 1)
    t1 = time.time()

    sz_hw = h * w
    sz_hwa = sz_hw * num_anchors
    t2 = time.time()
    for b in range(batch):
        boxes = []
        for cy in range(h):
            for cx in range(w):
                for i in range(num_anchors):
                    ind = b * sz_hwa + i * sz_hw + cy * w + cx
                    det_conf = det_confs[ind]
                    if only_objectness:
                        conf = det_confs[ind]
                    else:
                        conf = det_confs[ind] * cls_max_confs[ind]

                    if conf > conf_thresh:
                        bcx = xs[ind]
                        bcy = ys[ind]
                        bw = ws[ind]
                        bh = hs[ind]
                        cls_max_conf = cls_max_confs[ind]
                        cls_max_id = cls_max_ids[ind]
                        box = [bcx / w, bcy / h, bw / w, bh / h, det_conf, cls_max_conf, cls_max_id]
                        if (not only_objectness) and validation:
                            for c in range(num_classes):
                                tmp_conf = cls_confs[ind][c]
                                if c != cls_max_id and det_confs[ind] * tmp_conf > conf_thresh:
                                    box.append(tmp_conf)
                                    box.append(c)
                        if box[3] < 0.36 and box[2] < 0.3: ## Threshold to discard bigger people
                            boxes.append(box)

        all_boxes.append(boxes)
    t3 = time.time()
    if False:
        print('---------------------------------')
        print('matrix computation : %f' % (t1 - t0))
        print('        gpu to cpu : %f' % (t2 - t1))
        print('      boxes filter : %f' % (t3 - t2))
        print('---------------------------------')
    return all_boxes


def plot_boxes_cv2(imgO, boxes, savename=None, class_names=None, color=None):
    img = imgO.copy()
    #import cv2
    colors = FloatTensor([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]]);

    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

    width = img.shape[1]
    height = img.shape[0]
    boxes_ = []
    bbox = []
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = int((box[0] - box[2] / 2.0) * width)
        y1 = int((box[1] - box[3] / 2.0) * height)
        x2 = int((box[0] + box[2] / 2.0) * width)
        y2 = int((box[1] + box[3] / 2.0) * height)

        if color:
            rgb = color
        else:
            rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            if cls_id == 0 :
                #print('%s: %f' % (class_names[cls_id], cls_conf))
                # classes = len(class_names)
                # offset = cls_id * 123457 % classes
                # red = get_color(2, offset, classes)
                # green = get_color(1, offset, classes)
                # blue = get_color(0, offset, classes)
                # if color is None:
                #     rgb = (red, green, blue)
                # #txt = class_names[cls_id] + ' ' + str(cls_conf)
                # txt = '%0.3f'%cls_conf
                #img = cv2.putText(img, class_names[cls_id], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, rgb, 1)
                #img = cv2.putText(img, txt, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, rgb, 1)
                #img = cv2.rectangle(img, (x1, y1), (x2, y2), RGB, 1)
                img = cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
                #img = cv2.circle(img, (x1,y1), 2, (255,255,255), 2)
                boxes_.append(
                    (int(np.abs(x2-x1)/2.+x1),
                    int(np.abs(y2-y1)/2.+y1))
                    )
                bbox.append(((x1,x2),(y1,y2)))
    if savename:
        print("save plot results to %s" % savename)
        cv2.imwrite(savename, img)
    return img,boxes_, bbox 


def plot_boxes(img, boxes, savename=None, class_names=None):
    colors = FloatTensor([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]]);

    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

    width = img.width
    height = img.height
    draw = ImageDraw.Draw(img)
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = (box[0] - box[2] / 2.0) * width
        y1 = (box[1] - box[3] / 2.0) * height
        x2 = (box[0] + box[2] / 2.0) * width
        y2 = (box[1] + box[3] / 2.0) * height

        rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue = get_color(0, offset, classes)
            rgb = (red, green, blue)
            draw.text((x1, y1), class_names[cls_id], fill=rgb)
        draw.rectangle([x1, y1, x2, y2], outline=rgb)
    if savename:
        print("save plot results to %s" % savename)
        img.save(savename)
    return img


def read_truths(lab_path):
    if not os.path.exists(lab_path):
        return np.array([])
    if os.path.getsize(lab_path):
        truths = np.loadtxt(lab_path)
        truths = truths.reshape(truths.size / 5, 5)  # to avoid single truth problem
        return truths
    else:
        return np.array([])


def read_truths_args(lab_path, min_box_scale):
    truths = read_truths(lab_path)
    new_truths = []
    for i in range(truths.shape[0]):
        if truths[i][3] < min_box_scale:
            continue
        new_truths.append([truths[i][0], truths[i][1], truths[i][2], truths[i][3], truths[i][4]])
    return np.array(new_truths)


def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names


def image2torch(img):
    width = img.width
    height = img.height
    img = ByteTensor(ByteStorage.from_buffer(img.tobytes()))
    img = img.view(height, width, 3).transpose(0, 1).transpose(0, 2).contiguous()
    img = img.view(1, 3, height, width)
    img = img.float().div(255.0)
    return img


def do_detect(model, img, conf_thresh, nms_thresh, use_cuda=True):
    #model.eval()
    t0 = time.time()

    if isinstance(img, Image.Image):
        width = img.width
        height = img.height
        img = ByteTensor(ByteStorage.from_buffer(img.tobytes()))
        img = img.view(height, width, 3).transpose(0, 1).transpose(0, 2).contiguous()
        img = img.view(1, 3, height, width)
        img = img.float().div(255.0)
    elif type(img) == np.ndarray:  # cv2 image
        img = from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    else:
        print("unknow image type")
        exit(-1)

    t1 = time.time()

    if use_cuda:
        img = img.cuda()
    img = Variable(img)
    t2 = time.time()

    list_boxes = model(img)

    anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
    num_anchors = 9
    anchor_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    strides = [8, 16, 32]
    anchor_step = len(anchors) // num_anchors
    boxes = []
    for i in range(3):
        masked_anchors = []
        for m in anchor_masks[i]:
            masked_anchors += anchors[m * anchor_step:(m + 1) * anchor_step]
        masked_anchors = [anchor / strides[i] for anchor in masked_anchors]
        if use_cuda:
            boxes.append(get_region_boxes1(list_boxes[i].cpu().data.numpy(), conf_thresh, 80, masked_anchors, len(anchor_masks[i])))
        else:
            boxes.append(get_region_boxes1(list_boxes[i].data.numpy(), conf_thresh, 80, masked_anchors, len(anchor_masks[i])))
        # boxes.append(get_region_boxes(list_boxes[i], 0.6, 80, masked_anchors, len(anchor_masks[i])))

    boxes = boxes[0][0] + boxes[1][0] + boxes[2][0]
    t3 = time.time()

    boxes = nms(boxes, nms_thresh)
    t4 = time.time()

    if False:
        print('-----------------------------------')
        print(' image to tensor : %f' % (t1 - t0))
        print('  tensor to cuda : %f' % (t2 - t1))
        print('         predict : %f' % (t3 - t2))
        print('             nms : %f' % (t4 - t3))
        print('           total : %f' % (t4 - t0))
        print('-----------------------------------')
    return boxes


def read_data_cfg(datacfg):
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(datacfg, 'r') as fp:
        lines = fp.readlines()

    for line in lines:
        line = line.strip()
        if line == '':
            continue
        key, value = line.split('=')
        key = key.strip()
        value = value.strip()
        options[key] = value
    return options


def scale_bboxes(bboxes, width, height):
    import copy
    dets = copy.deepcopy(bboxes)
    for i in range(len(dets)):
        dets[i][0] = dets[i][0] * width
        dets[i][1] = dets[i][1] * height
        dets[i][2] = dets[i][2] * width
        dets[i][3] = dets[i][3] * height
    return dets


def file_lines(thefilepath):
    count = 0
    thefile = open(thefilepath, 'rb')
    while True:
        buffer = thefile.read(8192 * 1024)
        if not buffer:
            break
        count += buffer.count('\n')
    thefile.close()
    return count


def get_image_size(fname):
    '''Determine the image type of fhandle and return its size.
    from draco'''
    with open(fname, 'rb') as fhandle:
        head = fhandle.read(24)
        if len(head) != 24:
            return
        if imghdr.what(fname) == 'png':
            check = struct.unpack('>i', head[4:8])[0]
            if check != 0x0d0a1a0a:
                return
            width, height = struct.unpack('>ii', head[16:24])
        elif imghdr.what(fname) == 'gif':
            width, height = struct.unpack('<HH', head[6:10])
        elif imghdr.what(fname) == 'jpeg' or imghdr.what(fname) == 'jpg':
            try:
                fhandle.seek(0)  # Read 0xff next
                size = 2
                ftype = 0
                while not 0xc0 <= ftype <= 0xcf:
                    fhandle.seek(size, 1)
                    byte = fhandle.read(1)
                    while ord(byte) == 0xff:
                        byte = fhandle.read(1)
                    ftype = ord(byte)
                    size = struct.unpack('>H', fhandle.read(2))[0] - 2
                    # We are at a SOFn block
                fhandle.seek(1, 1)  # Skip `precision' byte.
                height, width = struct.unpack('>HH', fhandle.read(4))
            except Exception:  # IGNORE:W0703
                return
        else:
            return
        return width, height


def logging(message):
    print('%s %s' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), message))


#=====MASK RCNN

import cv2
import numpy as np
import random
#from transforms.transforms import *
# this will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

def get_outputs(image, model, threshold,device):
    with no_grad():
        image = from_numpy(image.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
        if device:
            image = image.cuda()
        # forward pass of the image through the modle
        outputs = model(image)
    
    # get all the scores
    scores = list(outputs[0]['scores'].detach().cpu().numpy())
    # index of those scores which are above a certain threshold
    thresholded_preds_inidices = [scores.index(i) for i in scores if i > threshold]
    thresholded_preds_count = len(thresholded_preds_inidices)
    # get the masks
    masks = (outputs[0]['masks']>0.5).squeeze().detach().cpu().numpy()
    # discard masks for objects which are below threshold
    masks = masks[:thresholded_preds_count]
    # get the bounding boxes, in (x1, y1), (x2, y2) format
    boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]  for i in outputs[0]['boxes'].detach().cpu()]
    # discard bounding boxes below threshold value
    boxes = boxes[:thresholded_preds_count]
    # get the classes labels
    labels = [coco_names[i] for i in outputs[0]['labels']]
    labels = labels[:thresholded_preds_count]
    #print(labels)
    return masks, boxes, labels

def filter_outputs(masks,boxes,labels,labeloi='person'):
    masks_ = []
    boxes_ = []
    labels_ = []
    for i in range(len(masks)):
        #print(masks[i])
        #print(labels[i])
        if labels[i] == labeloi:
            masks_.append(masks[i])
            boxes_.append(boxes[i])
            labels_.append(labels[i])
    
    return masks_,boxes_,labels_

# define the torchvision image transforms
transform = transforms.Compose([
    transforms.ToTensor(),
])

def detect_predict(image,model,device=True,threshold=0.65):
    # transform the image to tensor
    with no_grad():
        image = from_numpy(image.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
        if device:
            image = image.cuda()
    outputs = model(image) # get the predictions on the image

    # get score for all the predicted objects
    scores = list(outputs[0]['scores'].detach().cpu().numpy())
    # index of those scores which are above a certain threshold
    thresholded_preds_inidices = [scores.index(i) for i in scores if i > threshold]
    thresholded_preds_count = len(thresholded_preds_inidices)

    # get all the predicted bounding boxes
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    # get boxes above the threshold score
    #boxes = pred_bboxes[pred_bboxes > detection_threshold]
    boxes = pred_bboxes[:thresholded_preds_count]
    labels = outputs[0]['labels'].detach().cpu()
    labels = labels[:thresholded_preds_count]

    # get all the predicited class names
    pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]
    pred_classes = pred_classes[:thresholded_preds_count]
    
    return pred_classes, boxes, labels#outputs[0]['labels']


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
# the function takes the original prediction and the iou threshold.

def apply_nms(orig_prediction, iou_thresh=0.3):

    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)

    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]

    return final_prediction