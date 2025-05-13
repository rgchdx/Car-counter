from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *


cap = cv2.VideoCapture('/Users/rgdix/Desktop/object_detection/car_counter/Videos/cars.mp4')  # Load a video file
cap.set(3, 1280)  # Set width which is 3rd index
cap.set(4, 720)  # Set height which is 4th index

# Creating YOLO model
model = YOLO('yolov8n.pt')  # Load the YOLOv8 model. You can also use yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt

class_names = model.names  # Get the class names from the model
# if this above does not work
# class_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket"]
bound = cv2.imread('bound.png')


while True:
    success, img = cap.read()
    # the region of interest is the area where the car will be detected
    imgRegion = cv2.bitwise_and(img, bound)
    if not success:
        print("End of video")
        break
    results = model(imgRegion, stream=True)  # Get the results from the model. Stream=True is used to get the results in a stream which is faster
    for r in results:
        boxes = r.boxes # Get the bounding boxes which are the detected objects
        for box in boxes:
            # for openCV
            x1,y1,x2,y2 = box.xyxy[0] # Get the coordinates of the bounding box. Default is tensor output
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2) # Convert the coordinates to integers
            #cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255), 2)
            
            # print(x1,y1,x2,y2)
            # args are (x1,y1) and (x2,y2) which are the coordinates of the bounding box
            # then the color and  then thickness of the rectangle
            # for cvzone
            w, h = x2-x1, y2-y1
            cvzone.cornerRect(img, (x1, y1, w, h), l=8) # Draw the rectangle on the image
            
            # these are the confidence values for the detected objects. Higher the better
            conf = math.ceil((box.conf[0]*100))/100 # Get the confidence value of the detected object. Default is tensor output
            print(conf)
            
            # Class name
            cls = int(box.cls[0])
            currentClass = class_names[cls]
            if currentClass == "car" or currentClass == "motorbike" or currentClass == "bus" or currentClass == "truck" and conf>0.3: #only display confidence vals for cars
                # this is a box that will contain the text of the confidence.
                # If the box goes out of the camera area, it will be shows below the camera area
                cvzone.putTextRect(img, f'{class_names[cls]} {conf}', (max(0, x1), max(35, y1)), scale=0.6, thickness=1, offset=3)
            
            
    cv2.imshow("Image", img) # Display the image
    cv2.imshow("Region", imgRegion) # Display the region of interest
    cv2.waitKey(1) # Wait for 1 ms