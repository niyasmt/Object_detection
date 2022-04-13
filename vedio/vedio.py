import cv2 as cv

cap = cv.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 470)
cap.set(10, 70)

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

font_scale = 3
font = cv.FONT_HERSHEY_PLAIN

while True:
    ret, frame = cap.read()
    
    classindex, confidence, bbox = model.detect(frame, confThreshold = 0.5)
    

    if len(classindex) != 0:
        for classind, conf, boxes in zip(classindex.flatten(), confidence.flatten(), bbox):
            if (classind <= 80):
                cv.rectangle(frame, boxes, (255, 0, 0), 2)
                print(classind -1)
                print(classlabels[classind - 1])
                cv.putText(frame, classlabels[classind - 1], (boxes[0] + 10, boxes[1] + 30), font, font_scale, (0,255,0), 3)

    cv.imshow('object detection', frame)
    if cv.waitKey(2) & 0xff == ord('q'):
        break

cap.release()
cv.destroyAllWindows()