import cv2
import numpy as np


# Web Camera
cap = cv2.VideoCapture('video.mp4')

# Initialize Subtractor
algo = cv2.bgsegm.createBackgroundSubtractorMOG()

while True:
    ret,frame1 = cap.read()  # Capture frame-by-frame
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    blur = cv2.GaussianBlur(grey, (3,3), 5)  # Apply Gaussian blur

    # Apply background subtraction
    img_sub = algo.apply(blur)

    # Post-processing
    dilat = cv2.dilate(img_sub, np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)

    # Find contours
    counters = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Draw contours on frame
    frame_result = cv2.drawContours(frame1, counters, -1, (0, 255, 0), 2)
    # Display frames
    cv2.imshow('frame', frame_result)
    cv2.imshow('Detector', dilatada)
    cv2.imshow('Video Original', frame1)
    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()

