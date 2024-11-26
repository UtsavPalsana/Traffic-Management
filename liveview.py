import cv2

capture=cv2.VideoCapture("http://192.168.1.6:8080/video")

while(True):
    _, frame = capture.read()
    cv2.imshow('liveview', frame)

    if cv2.waitKey(10) & 0xff == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()    
