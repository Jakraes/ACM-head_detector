import cv2
import numpy as np
import time

cv2.setUseOptimized(True)

cap = cv2.VideoCapture("./videos/test5.mp4")

with open("obj.names", 'rt') as f:
    labels = f.read().rstrip('\n').split('\n')
weights = "yolov4-tiny-custom_best.weights"
config = "yolov4-tiny-custom.cfg"

_confidence = 0.5
_threshold = 0.3

np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet(config, weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

last_boxes = []

while cap.isOpened():
    start = time.time()
    _, frame = cap.read()

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)

    layerOutputs = net.forward(ln)

    boxes, confidences, classIDs = [], [], []

    for output in layerOutputs:
        for detection in output:

            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > _confidence:

                box = np.array([detection[0] * frame.shape[1], detection[1] * frame.shape[0],
                               detection[2] * frame.shape[1], detection[3] * frame.shape[0]])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, _confidence,
                            _threshold)

    if len(idxs) > 0:
        for i in idxs.flatten():

            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in colors[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
            cv2.putText(frame, text + f" - {i}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

    key = cv2.waitKey(1)
    if key == 27:
        break

    end = time.time()
    cv2.putText(frame, f"{1/(end - start)}", (5, 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 0, 0), 2)

    cv2.imshow("Test", frame)

cv2.destroyAllWindows()
