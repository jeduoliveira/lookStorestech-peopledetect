import cv2
import time

captura = cv2.VideoCapture("tcp://192.168.0.118:8888")

frame_num = 0
while(1):
    ret, frame = captura.read()
    #cv2.imshow("Video", frame)
   
    frame_num +=1
    print('Frame #: ', frame_num)
    start_time = time.time()

    fps = 1.0 / (time.time() - start_time)
    print("FPS: %.2f" % fps)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

captura.release()
cv2.destroyAllWindows()