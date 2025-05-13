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

# Tracking
tracker = Sort(max_age=20, min_hits=2, iou_threshold=0.3) # Create a tracker object. max_age is the maximum age of a track before it is deleted. min_hits is the minimum number of hits before a track is considered valid. iou_threshold is the intersection over union threshold for the tracker

# Line to increment the count when crossed
line = [(0, 500), (1280, 500)] # This is the line that will be used to increment the count when crossed
totalCount = []

while True:
    success, img = cap.read()
    # the region of interest is the area where the car will be detected
    imgRegion = cv2.bitwise_and(img, bound)
    if not success:
        print("End of video")
        break
    results = model(imgRegion, stream=True)  # Get the results from the model. Stream=True is used to get the results in a stream which is faster
    detections = np.empty((0, 5))
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
                cvzone.cornerRect(img, (x1, y1, w, h), l=8,rt=5) # Draw the rectangle on the image
                currentArray = np.array([x1, y1, x2, y2, conf]) # 5 values that are required for the tracker
                detections =  np.vstack((detections, currentArray)) # Stack the current array to the detections array. This is used to get the coordinates of the bounding box and the confidence value
    # Tracking
    resultTracker = tracker.update(detections) # Update the tracker with the new detections
    
    cv2.line(img, line[0], line[1], (0, 255, 0), 3) # Draw the line on the image
    
    for result in resultTracker:
        x1,y1,x2,y2,id = result
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        print(result)
        cvzone.cornerRect(img, (x1,y1,w,h),l=8,rt=2,color=(255,0,0))
        cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)
        # line crossing
        cx,cy, = x1+w//2, y1+h//2 # Get the center of the bounding box
        # this is the center of the bounding box. draw circle here
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        if line[0][0] < cx < line[1][0] and line[1][0]-30 < cy < line[1][1]+30:
            # make this a list so that we detect only once
            if totalCount.count(id) == 0:
                totalCount.append(id)
                # overlay to show that it was detected
                cvzone.cornerRect(img, (x1,y1,w,h),l=8,rt=2,color=(255,255,0))
            
    cvzone.putTextRect(img, f'Count: {len(totalCount)}', (50,50))
    cv2.imshow("Image", img) # Display the image
    cv2.imshow("Region", imgRegion) # Display the region of interest
    cv2.waitKey(1) # Wait for 1 ms