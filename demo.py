import cv2
import numpy as np
import os
import argparse
import pickle
import time
import imutils
from imutils.video import VideoStream

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from PIL import Image

from Module.mobilenetv2 import*

image_valid_transforms = transforms.Compose([
    transforms.Resize(size=78),
    transforms.CenterCrop(size=64),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=False,
                help="path to trained model")
ap.add_argument("-l", "--le", type=str, required=False,
                help="path to label encoder", default="1e-4")
ap.add_argument("-d", "--detector", type=str, required=False,
                help="path to OpenCV's deep learning face detector", default="./face_detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())


print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args['detector'], "deploy.prototxt"])
# print(protoPath)
modelPath = os.path.sep.join([args["detector"],
                              "res10_300x300_ssd_iter_140000.caffemodel"])

net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load the liveness detector model and label encoder from disk
print("[INFO] loading face detector...")

model = MobileNetV2()

model.load_state_dict(torch.load('./checkpoint/mobilenetv2_v0.0.2_BS_64_IS_64_Adam.pkl',
                                 map_location=torch.device('cpu')))

model.eval()
############################################

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream")

vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 600 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=800)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]
        # print(confidence)

        # filter out weak detections
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for
            # the face and extract the face ROI
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the detected bounding box does fall outside the
            # dimensions of the frame
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            # extract the face ROI and then preproces it in the exact
            # same manner as our training data
            face = frame[startY:endY, startX:endX]

            face = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            # face.save('{}.png'.format(i))

            face = image_valid_transforms(face).unsqueeze_(0)

            # pass the face ROI through the trained liveness detector
            # model to determine if the face is "real" or "fake"
            preds = model(face)
            _, indx = torch.max(preds, dim=1)
            prob = F.softmax(preds)
            indx = indx[0].numpy()
            if indx == 0:
                label = 'fake'
            if indx == 1:
                label = 'real'

            # draw the label and bounding box on the frame
            label = "{}:{:.4f}".format(label, prob[0][indx].detach().numpy())
            cv2.putText(frame, label, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)


    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
