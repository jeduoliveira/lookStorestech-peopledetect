
apt update -y
apt upgrade -y 
apt install git -y
apt full-upgrade -y

apt install python3-dev git awscli -y
apt install -y python3 python3-pip
apt install libffi-dev libssl-dev -y
apt install python3-dev -y
apt install -y python3 python3-pip

git clone https://github.com/jeduoliveira/lookStorestech-peopledetect.gi 

cd lookStorestech-peopledetect

baixar o arquivo yolov4.weights em  https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT e salvar em ./data/

python3 -m venv .venv 
source .venv/bin/activate
pip install -r requirements.txt

python3 save_model.py --model yolov4 
python main.py --video 0  --model yolov4  