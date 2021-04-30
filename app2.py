import cv2
import numpy as np

net = cv2.dnn.readNet('yolov3_training_last.weights', 'yolov3_testing.cfg')
with open("classes.txt", "r") as f:
    classes = f.read().splitlines()
print(classes)

layer_names=net.getLayerNames()
outputlayers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
print(outputlayers)

img=cv2.imread("test5.jpg")
img=cv2.resize(img,None,fx=0.8,fy=0.6)
height, width, channels=img.shape

blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0,0,0), True, crop=False)
for b in blob:
    for n, img_blob in enumerate(b):
        cv2.imshow(str(n), img_blob)

net.setInput(blob)
outs=net.forward(outputlayers)


boxes = []
confidences = []
class_ids = []

for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.62:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)
                cv2.circle(img,(center_x,center_y),10,(0,255,0),2)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.6)
colors = np.random.uniform(0, 255, size=(100, 3))
font=cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x,y,w,h=boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[i]
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        cv2.putText(img, label, (x, y+30), font, 1, (255,255,255), 2)

cv2.imshow('Image', img)
cv2.waitKey(10000)

cv2.destroyAllWindows()
