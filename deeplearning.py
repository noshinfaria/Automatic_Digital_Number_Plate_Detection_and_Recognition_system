import os
import sys
# sys.path.append('/yolov7')


import argparse
import time
from pathlib import Path
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from numpy import random
import PIL
#from google.colab.patches import cv2_imshow
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, xywh2xyxy, clip_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
import functools
import tensorflow as tf

model_vgg19 = tf.keras.models.load_model('./static/models/model_vgg19.h5')

#number plate detection------------------------------------------------------------------------

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)



classes_to_filter = None  #You can give list of classes to filter by name, Be happy you don't have to put class number. ['train','person' ]
opt  = {
    
    "weights": "last.pt", # Path to weights file default weights are for nano model
    "yaml": "data.yaml",
    "img-size": 640, # default image size
    "conf-thres": 0.5, # confidence threshold for inference.
    "iou-thres" : 0.45, # NMS IoU threshold for inference.
    "device" : 'cpu',  # device to run our model i.e. 0 or 0,1,2,3 or cpu
    "classes" : classes_to_filter  # list of classes to filter or None

}


def plate_detection_i(source_image_path, filename):

    with torch.no_grad():
        weights, imgsz = opt['weights'], opt['img-size']
        set_logging()
        device = select_device(opt['device'])
        half = device.type != 'cpu'
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if half:
            model.half()

        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

        img0 = cv2.imread(source_image_path)
        img = letterbox(img0, imgsz, stride=stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=False)[0]

        # Apply NMS
        classes = None
        if opt['classes']:
            classes = []
            for class_name in opt['classes']:
                classes.append(opt['classes'].index(class_name))

        pred = non_max_suppression(pred, opt['conf-thres'], opt['iou-thres'], classes=classes, agnostic=False)
        t2 = time_synchronized()
        for i, det in enumerate(pred):
            s = ''
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                for *xyxy, conf, cls in reversed(det):

                    label = f'{names[int(cls)]} {conf:.2f}'
                    # plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)

                    #to save
                    BGR = True
                    xyxy = torch.tensor(xyxy).view(-1, 4)
                    b = xyxy2xywh(xyxy)  # boxes
                    b[:, 2:] = b[:, 2:] * 1.02 + 10  # box wh * gain + pad ;gain=1.02, pad=10
                    xyxy = xywh2xyxy(b).long()
                    clip_coords(xyxy, img0.shape)
                    crop = img0[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::(1 if BGR else -1)]

                    # check aspect ratio to reduce gerbage

                    ar = 1.0 * (crop.shape[1] / crop.shape[0])  # w/h; approximate aspect ration for number plate is 1.8
                    print(ar)
                    if (ar > 1.3 and ar <= 2.6):
                        img_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                        img_bgr = cv2.resize(img_bgr, (300, 150))
                        cv2.imwrite('./static/plate_img/{}'.format(filename), img_bgr)
                        crop_img = img_bgr
                        # print(ar)
                    else:
                        crop_img = img0

    return crop_img



#--------- character localization-----------------------------------------------------------

def character_localize_i(after_validation_img, filename):
    ar = 1.0 * (after_validation_img.shape[1] / after_validation_img.shape[0])
    if (ar > 1.3 and ar <= 2.6):
        resize = cv2.resize(after_validation_img, (500, 250))
        # cv2_imshow(resize)

        ch_gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
        # cv2_imshow(ch_gray)

        # blurred = cv2.GaussianBlur(ch_gray, (5, 5), 0)
        # #cv2_imshow(blurred)

        # canny = cv2.Canny(blurred, 170,200)
        # #cv2_imshow(canny)
        thresh = cv2.adaptiveThreshold(ch_gray, 255,
                                       cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 29, 15)
        ###cv2_imshow(thresh)

        kernel = np.ones((3, 3), 'uint8')
        img_dilation = cv2.dilate(thresh, kernel, iterations=1)
        ###cv2_imshow(img_dilation)

        # There are many white “blobs” in the binary image.
        # We need to determine which white blobs are license plate characters.
        # This can be done by applying an algorithm called connected-component analysis.
        # Perform connected components analysis on the thresholded image and
        # initialize the mask to hold only the components we are interested in
        _, labels = cv2.connectedComponents(img_dilation)
        mask = np.zeros(thresh.shape, dtype="uint8")

        # The connectedComponents method returns labels, a NumPy array with the same dimension as our thresh image.
        # Each element in labels is 0 if it is background or >0 if it belongs to a connected-component.
        # Each connected-component corresponds a white blob and has a unique label.
        # So how can we decide if a white blob is of a character? An heuristic approach is used here.
        # From the binary image it can be seen that the number of pixels for every character falls in a certain range.
        # Therefore we set a lower boundary and an upper boundary which are the number of pixels within which each connected-component must have:
        # Set lower bound and upper bound criteria for characters
        total_pixels = after_validation_img.shape[0] * after_validation_img.shape[1]
        # print(total_pixels)
        lower = total_pixels // 100  # 100 heuristic param, can be fine tuned if necessary
        upper = total_pixels // 1  # 6 heuristic param, can be fine tuned if necessary
        # print(lower)
        # print(upper)

        # Loop over the unique components
        for (i, label) in enumerate(np.unique(labels)):
            # If this is the background label, ignore it
            if label == 0:
                continue

            # Otherwise, construct the label mask to display only connected component
            # for the current label
            labelMask = np.zeros(thresh.shape, dtype="uint8")
            labelMask[labels == label] = 255
            numPixels = cv2.countNonZero(labelMask)

            # If the number of pixels in the component is between lower bound and upper bound,
            # add it to our mask
            if numPixels > lower and numPixels < upper:
                mask = cv2.add(mask, labelMask)

        ###cv2_imshow(mask)

        # By finding contours we can get the bounding boxes of the license plate characters:
        # Find contours and get bounding box for each contour
        cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]

        # Sort the bounding boxes from left to right, top to bottom
        # sort by Y first, and then sort by X if Ys are similar
        def compare(rect1, rect2):
            if abs(rect1[1] - rect2[1]) > 10:
                return rect1[1] - rect2[1]
            else:
                return rect1[0] - rect2[0]

        boundingBox = sorted(boundingBoxes, key=functools.cmp_to_key(compare))

        # image with bounding box
        result = thresh.copy()
        cnt = 0
        crop_imgs = []
        for box in boundingBox:
            x, y, w, h = box
            # print(w, h) 35,34
            # for reducing garbage check ratio 45
            if (350 > w > 30 and 109 > h > 30):
                cnt += 1
                # if(99999 > w > 0 and 999999>h>45):
                crp_img = result[y:y + h, x:x + w]
                # cv2_imshow(crp_img)
                crop_imgs.append(crp_img)
                # path = '/content/drive/MyDrive/Colab_Notebooks/capstone/character_data'
                # cv2.imwrite(os.path.join(path, str(name)+ '.png'), crp_img)
                # name +=1
                cv2.rectangle(result, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (200, 200, 200), 2)
                only_for_save = cv2.resize(result, (300, 150))
                cv2.imwrite('./static/loc_char_img/{}'.format(filename), only_for_save)
        ###cv2_imshow(result)

        test = thresh.copy()
        if cnt == 0:
            max_w = np.max(boundingBoxes)
            for box in boundingBox:
                if box[2] == max_w:
                    x, y, w, h = box
                    crp_img2 = test[y + 10:y + h - 10, x + 20:x + w - 20]
                    # print(w)
            # cv2_imshow(crp_img2)
            crp_img2 = cv2.resize(crp_img2, (500, 250))

            # By finding contours, we can set the bounding boxes to license plate characters:
            # Find contours and get bounding box for each contour
            cnts, _ = cv2.findContours(crp_img2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boundingBoxes = [cv2.boundingRect(c) for c in cnts]

            # Sort the bounding boxes from left to right, top to bottom
            # sort by Y first, and then sort by X if Ys are similar
            def compare(rect1, rect2):
                if abs(rect1[1] - rect2[1]) > 10:
                    return rect1[1] - rect2[1]
                else:
                    return rect1[0] - rect2[0]

            boundingBox = sorted(boundingBoxes, key=functools.cmp_to_key(compare))

            # image with bounding box
            result = crp_img2.copy()
            cnt = 0
            for box in boundingBox:
                x, y, w, h = box
                # print(w, h)
                # for reducing garbage check ratio 45
                if (350 > w > 35 and 109 > h > 34):
                    cnt += 1
                    # if(99999 > w > 0 and 999999>h>45):
                    crp_img = result[y:y + h, x:x + w]
                    crop_imgs.append(crp_img)
                    # path = '/content/drive/MyDrive/Colab_Notebooks/capstone/character_data'
                    # cv2.imwrite(os.path.join(path, str(name)+ '.png'), crp_img)
                    # name +=1
                    cv2.rectangle(result, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (200, 200, 200), 2)
                    only_for_save = cv2.resize(result, (300, 150))
                    cv2.imwrite('./static/loc_char_img/{}'.format(filename), only_for_save)

        # for img in crop_imgs:
        #   cv2_imshow(img)

        # show thresh and result
        ###cv2_imshow(result)
        # cv2.imwrite('./static/{}'.format(filename), result)

    else:
        crop_imgs = after_validation_img
    return crop_imgs
#
# # source_image_path = '/content/gdrive/MyDrive/yolov7/IMG_0554.jpg'
# # crop = plate_detection(source_image_path)
# # cv2_imshow(crop)
#
#
#
#-------character recognition--------------------------------
def character_recognition_i(crop_imgs):
    if len(crop_imgs)!=1:
        # crop_plate = plate_detection(source_image_path, filename)
        # crop_imgs = character_localize(crop_plate)

        result = []
        for img in crop_imgs:
            img = cv2.merge([img, img, img])
            resize_img = cv2.resize(img, (224, 224))
            img = resize_img.reshape(1, 224, 224, 3)

            tags = {0: '০', 1: '১', 2: '২', 3: '৩', 4: '৪', 5: '৫', 6: '৬', 7: '৭', 8: '৮', 9: '৯',
                    10: 'ঢাকা', 11: 'মেট্রো', 12: 'খ', 13: 'গ', 14: 'চ', 15: 'ক', 16: 'স', 17: 'খালী',
                    18: 'পটুয়া', 19: 'পুর', 20: 'ম', 21: 'ভ', 22: 'গাজী', 23: 'ঠ'}

            pred = model_vgg19.predict(img)
            classIndex = tags[int(np.argmax(pred, axis=1))]
            probVal = np.amax(pred)

            #---- to reduce garbage ----------------------------------------
            if probVal > 0.50:
                result.append(classIndex)
                # print('prediction accuracy :',probVal)
                # print('predicted class :',classIndex)
        str1 = ' '.join([str(elem) for elem in result])

        print(str1)
        return str1

    else:
        return 0


#--------------------------detect from video--------------------
# video = cv2.VideoCapture(0)


# def detect_from_video():
#     with torch.no_grad():
#         weights, imgsz = opt['weights'], opt['img-size']
#         set_logging()
#         device = select_device(opt['device'])
#         half = device.type != 'cpu'
#         model = attempt_load(weights, map_location=device)  # load FP32 model
#         stride = int(model.stride.max())  # model stride
#         imgsz = check_img_size(imgsz, s=stride)  # check img_size
#         if half:
#             model.half()
#
#         names = model.module.names if hasattr(model, 'module') else model.names
#         colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
#         if device.type != 'cpu':
#             model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
#
#         cnt = 0
#         classes = None
#         if opt['classes']:
#             classes = []
#             for class_name in opt['classes']:
#                 classes.append(opt['classes'].index(class_name))
#
#         while True:
#
#             ret, img0 = video.read()
#             if not ret:
#                 break
#             elif ret:
#                 img = letterbox(img0, imgsz, stride=stride)[0]
#                 img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
#                 img = np.ascontiguousarray(img)
#                 img = torch.from_numpy(img).to(device)
#                 img = img.half() if half else img.float()  # uint8 to fp16/32
#                 img /= 255.0  # 0 - 255 to 0.0 - 1.0
#                 if img.ndimension() == 3:
#                     img = img.unsqueeze(0)
#
#                 # Inference
#                 t1 = time_synchronized()
#                 pred = model(img, augment=False)[0]
#
#                 pred = non_max_suppression(pred, opt['conf-thres'], opt['iou-thres'], classes=classes, agnostic=False)
#                 t2 = time_synchronized()
#                 for i, det in enumerate(pred):
#                     s = ''
#                     s += '%gx%g ' % img.shape[2:]  # print string
#                     gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
#                     if len(det):
#                         det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
#
#                         for c in det[:, -1].unique():
#                             n = (det[:, -1] == c).sum()  # detections per class
#                             s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
#
#                         for *xyxy, conf, cls in reversed(det):
#                             label = f'{names[int(cls)]} {conf:.2f}'
#
#                             # to save
#                             # cnt = 0
#                             if conf > 0.80 and cnt == 0:
#                                 BGR = True
#                                 xyxy = torch.tensor(xyxy).view(-1, 4)
#                                 b = xyxy2xywh(xyxy)  # boxes
#                                 b[:, 2:] = b[:, 2:] * 1.02 + 10  # box wh * gain + pad ;gain=1.02, pad=10
#                                 xyxy = xywh2xyxy(b).long()
#                                 clip_coords(xyxy, img0.shape)
#                                 crop = img0[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]),
#                                        ::(1 if BGR else -1)]
#
#                                 # check aspect ratio to reduce gerbage
#
#                                 ar = 1.0 * (crop.shape[1] / crop.shape[
#                                     0])  # w/h; approximate aspect ration for number plate is 1.8
#                                 print(ar)
#                                 if (ar > 1.3 and ar <= 2.6):
#                                     img_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
#                                     img_bgr = cv2.resize(img_bgr, (300, 150))
#                                     cv2.imwrite('./static/plate_img/{}'.format('hi.jpg'), img_bgr)
#                                     cnt += 1
#
#                                     # ---------start of localize character-------------------
#                                     loc_img = character_localize(img_bgr, 'filename')
#                                     recognition_char = character_recognition(loc_img)
#                                     print(recognition_char)
#
#                             # plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)
#
#                             # print(f"{j+1}/{nframes} frames processed")
#                             # output.write(img0)
#
#                 #---------end of detect number plate--------------------
#
#
#
#                 frame = cv2.imencode('.jpg', img0)[1].tobytes()
#                 yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#             # # time.sleep(0.1)
#             # key = cv2.waitKey(1) & 0xFF
#             # if key == ord("q"):
#             #     break
#     # cv2.destroyAllWindows()

def plate_from_video(img0, cnt):
    with torch.no_grad():
        weights, imgsz = opt['weights'], opt['img-size']
        set_logging()
        device = select_device(opt['device'])
        half = device.type != 'cpu'
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if half:
            model.half()

        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

        # cnt = 0
        classes = None
        if opt['classes']:
            classes = []
            for class_name in opt['classes']:
                classes.append(opt['classes'].index(class_name))

        img = letterbox(img0, imgsz, stride=stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=False)[0]

        pred = non_max_suppression(pred, opt['conf-thres'], opt['iou-thres'], classes=classes, agnostic=False)
        t2 = time_synchronized()
        for i, det in enumerate(pred):
            s = ''
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'

                    # to save
                    # cnt = 0
                    if conf > 0.80 and cnt == 0:
                        BGR = True
                        xyxy = torch.tensor(xyxy).view(-1, 4)
                        b = xyxy2xywh(xyxy)  # boxes
                        b[:, 2:] = b[:, 2:] * 1.02 + 10  # box wh * gain + pad ;gain=1.02, pad=10
                        xyxy = xywh2xyxy(b).long()
                        clip_coords(xyxy, img0.shape)
                        crop = img0[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]),
                               ::(1 if BGR else -1)]

                        # check aspect ratio to reduce gerbage

                        ar = 1.0 * (crop.shape[1] / crop.shape[
                            0])  # w/h; approximate aspect ration for number plate is 1.8
                        print(ar)
                        if (ar > 1.3 and ar <= 2.6):
                            img_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                            img_bgr = cv2.resize(img_bgr, (300, 150))
                            cv2.imwrite('./static/plate_img/{}'.format('hi.jpg'), img_bgr)
                            cnt += 1

                            # ---------start of localize character-------------------
                            loc_img = character_localize_i(img_bgr, 'filename')
                            recognition_char = character_recognition_i(loc_img)
                            print(recognition_char)
                            return recognition_char

                    # plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)

                    # print(f"{j+1}/{nframes} frames processed")
                    # output.write(img0)

        # ---------end of detect number plate--------------------


