import cv2
import numpy as np
import time

def nothing(x):
    pass

cap = cv2.VideoCapture(0) #webcam (video capture object)

cap.set(3,1280)
cap.set(4,720)
print(cap)


kernel = np.ones((5,5),np.uint8)    #5x5 kernel for morphological operations (dilation purpose)
canvas = None                       # Initializing the canvas

switch = 'Pen'

x1,y1=0,0 # Initilize x1,y1

noiseth = 800 #Threshold for noise

penc = [255,0,0]
thickness = 10

#------------------------------------------------MAIN CODE-------------------------------------------------------------#
while True:

    # ret: if the frame is available (boolean)
    # frame:  image array vector captured based on the default frames per second
    ret, frame = cap.read()
    print(ret)

    if not ret:
        break


    # flip frames
    frame = cv2.flip(frame, 1)

    k = cv2.waitKey(1) & 0xFF
    if  k == ord('p'):
        switch = 'Pen'
    if  k == ord('x'):
        switch = 'Eraser'

    #Initilize canvas as black image(same size as the frame)
    if canvas is None:
        canvas = np.zeros_like(frame)


    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #bgr to hsv conversion

    # bottle's color
#    lower_range = np.array([26, 80, 0])
#    upper_range = np.array([96, 255, 223])
    
#    trial pointer
    lower_range = np.array([40, 40, 40])
    upper_range = np.array([70,255,255])

    # pen's color
    #(136,100,90)
    #[52,127,153][107,255,255]
    # lower_range = np.array([50,55,45])
    # upper_range = np.array([130, 255, 255])
    # lower_range = np.array([52, 127, 153])
    # upper_range = np.array([107, 255, 255])
    # lower_range = np.array([19, 56, 97])
    # upper_range = np.array([107, 255, 255])
    mask = cv2.inRange(hsv, lower_range, upper_range)

    # morphological operations to get rid of the noise: Erosion eats the white part while dilation expands it.
    # kernel is overlapped over the image to calculate maximum pixel values
    # erode: pixel value calculated minimum, regions of darker shades increase
    #dilate: pixel value calculated maximum, increases in the white shade
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    #FindContours in the mask frame.
    #cv2.RETR_EXTERNAL: retrieves extreme outer contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Make sure there is a contour present and also its size is bigger than the noise threshold.
    if contours and cv2.contourArea(max(contours, key=cv2.contourArea)) > noiseth:
        c = max(contours, key=cv2.contourArea)
        x2, y2, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c) # Get the area of the contour

        # If there were no previous points then save the detected x2,y2 as x1,y1. This is true when:
        # 1. We're writing for the first time
        # 2. Writing again when the pen had disapeared from view.
        if x1 == 0 and y1 == 0:
            x1, y1 = x2, y2

        else:
            if switch == 'Pen':
                #,lineType=cv2.LINE_AA
                canvas = cv2.line(canvas, (x1, y1), (x2, y2), penc, thickness) # Draw the line on the canvas #10
            else:
                cv2.circle(canvas, (x2, y2), 30, (0, 0, 0), -1) 
        x1, y1 = x2, y2  # After the line is drawn the new points become the previous points.

    else:
        x1, y1 = 0, 0 # If there were no contours detected then make x1,y1 = 0 (out of view)



    frame = cv2.add(frame, canvas) 
    stacked = np.hstack((canvas, frame))
    cv2.imshow('Air Canvas', cv2.resize(stacked, None, fx=0.5, fy=0.6))  # Show this stacked frame at 40% of the size.

    k = cv2.waitKey(1) & 0xFF
    # press e to exit
    if k == ord('e'):
        break

    # press c to clear canvas
    if k == ord('c'):
        canvas = None

# Release the camera & destroy the windows.
cap.release()
cv2.destroyAllWindows()

#------------------------------------------  GUIDE  --------------------------------------------------------#

# Press c : clear
# Press e : EXIT

#Press p : pen
#Press x : eraser

#------------------------------------------------------------------------------------------------------------#
