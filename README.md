python3 setup.py bdist_wheel


https://downloads.raspberrypi.org/raspios_arm64/images/raspios_arm64-2021-05-28/2021-05-07-raspios-buster-arm64.zip


python main.py --video 0  --model yolov4





libcamera-vid -t 0 --width 1080 --height 720 -q 100 -n --codec mjpeg --inline --listen -o tcp://192.168.0.118:8888 -v