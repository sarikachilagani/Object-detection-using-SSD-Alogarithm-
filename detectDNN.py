import cv2
import time
import imutils
import numpy as np
import systemcheck
from imutils.video import VideoStream

threshold = 0.5
# Initialize Objects and corresponding colors which the model can detect
labels = ["background", "aeroplane", "bicycle", "bird",
          "boat", "bottle", "bus", "car", "cat", "chair", "cow",
          "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
          "sheep", "sofa", "train", "tvmonitor"]
colors = np.random.uniform(0, 255, size=(len(labels), 3))

# Loading Caffe Model
print('[Status] Loading Model...')
neural_network = cv2.dnn.readNetFromCaffe("./Caffe/SSD_MobileNet_prototxt.txt", "./Caffe/SSD_MobileNet.caffemodel")

# Initialize Video Stream
cam = VideoStream(src=0).start()
time.sleep(2.0)

# Loop Video Stream
while True:

    img = cam.read()
    img = imutils.resize(img, width=400)
    (h, w) = img.shape[:2]

    # Converting Frame to Blob
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)),
                                 0.007843, (300, 300), 127.5)

    # Passing Blob through network to detect and predict
    neural_network.setInput(blob)
    detections = neural_network.forward()

    # Loop over the detections
    for i in np.arange(0, detections.shape[2]):

        # Extracting the confidence of predictions
        confidence = detections[0, 0, i, 2]

        # Filtering out weak predictions
        if confidence > threshold:
            # Extracting the index of the labels from the detection
            # Computing the (x,y) - coordinates of the bounding box
            idx = int(detections[0, 0, i, 1])

            # Extracting bounding box coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Drawing the prediction and bounding box
            label = "{}: {:.2f}%".format(labels[idx], confidence * 100)
            cv2.rectangle(img, (startX, startY), (endX, endY), colors[idx], 2)

            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(img, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)

    cv2.imshow("detector", img)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

cv2.destroyAllWindows()
cam.stop()
