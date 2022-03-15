import cv2
import time
import numpy as np
from PIL import Image

vid = cv2.VideoCapture("tcp://192.168.0.118:8888")

frame_num = 0
while(1):
    return_value, frame = vid.read()
    if return_value:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
    else:
        print('Video has ended or failed, try a different video format!')
        break
    frame_num +=1
    print('Frame #: ', frame_num)
    frame_size = frame.shape[:2]
    image_data = cv2.resize(frame, (416, 416))
    image_data = image_data / 255.
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    start_time = time.time()
    fps = 1.0 / (time.time() - start_time)
    print("FPS: %.2f" % fps)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

vid.release()
cv2.destroyAllWindows()