import cv2 as cv
import matplotlib.pyplot as plt


classlabels = []
file_name = 'label.txt'
with open(file_name, 'rt') as fpt:
    classlabels = fpt.read().rstrip('\n').split('\n')

confiq_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'

model = cv.dnn_DetectionModel(frozen_model, confiq_file)


model.setInputSize(320,320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

print(len(classlabels))
print(classlabels)

print(classlabels[73])

image = cv.imread('photos/banana.jpg')


classIndex, confidence, bbox = model.detect(image, confThreshold = 0.5)
print(classIndex,',',  confidence,',',  bbox)

font_scale = 3
font = cv.FONT_HERSHEY_PLAIN
for classInd, conf, boxes in zip(classIndex.flatten(), confidence.flatten(), bbox):
    cv.rectangle(image, boxes, (255, 0, 0), 2)
    print(classInd - 1)
    print(classlabels[classInd - 1])
    cv.putText(image, classlabels[classInd - 1], (boxes[0] + 10, boxes[1] + 40), font, font_scale, (0, 255, 0), 3)

image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
plt.imshow(image)
plt.show()