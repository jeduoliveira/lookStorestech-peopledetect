import cv2


captura = cv2.VideoCapture("tcp://192.168.0.118:8888")

while(1):
    ret, frame = captura.read()
    cv2.imshow("Video", frame)
   
    if cv2.waitKey(1) & 0xFF == ord('q'): break

captura.release()
cv2.destroyAllWindows()