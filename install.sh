
#!/bin/bash
set -e

get_distribution() {
	lsb_dist=""
	# Every system that we officially support has /etc/os-release
	if [ -r /etc/os-release ]; then
		lsb_dist="$(. /etc/os-release && echo "$ID")"
	fi
	# Returning an empty string here should be alright since the
	# case statements don't act unless you provide an actual value
	echo "$lsb_dist"
}

do_install() {
	echo "# Iniciando a execucao do script"
    user="$(id -un 2 || true)"
	echo $user
	sh_c='sh -c'
	#if [ "$user" != 'root' ]; then
	#		cat >&2 <<-'EOF'
	#		Error: this installer needs the ability to run commands as root.
	#		We are unable to find either "sudo" or "su" available to make this happen.
	#		EOF
	#		exit 1  
	#fi

    lsb_dist=$( get_distribution )
	lsb_dist="$(echo "$lsb_dist" | tr '[:upper:]' '[:lower:]')"

    case "$lsb_dist" in
		ubuntu)
			apt-get -y update --allow-releaseinfo-change
			apt-get -y full-upgrade 
			apt-get -y autoremove
	
			apt-get -y install git awscli python3-pip python3-dev curl

			cd /opt
			if [ ! -d "/opt/lookStorestech-peopledetect" ] 
			then
				echo "# Clonando o projeto"
				git clone https://github.com/jeduoliveira/lookStorestech-peopledetect.git
			fi
			
			cd /opt/lookStorestech-peopledetect
			if [ ! -d "./.venv" ] 
			then
				 python3 -m pip install virtualenv
				 python3 -m virtualenv .venv

			fi

			. .venv/bin/activate

			pip3 install -U six wheel mock

			curl -L https://lookstoretech-frontend.s3.amazonaws.com/yolov4.weights -o ./data/yolov4.weights
			curl -L https://github.com/KumaTea/tensorflow-aarch64/releases/download/v2.7/tensorflow-2.7.0-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl -o tensorflow-2.7.0-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
			chmod +x tensorflow-2.7.0-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
			pip3 install tensorflow-2.7.0-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
			pip3 install -r requirements.txt 
			python3 save_model.py --model yolov4 

		;;
        debian|raspbian)

			echo "**** Realizando update"
			apt-get -y update > /dev/null
			apt-get -y full-upgrade 
			rpi-update -y

			echo "# Instalando pacotes necessarios para o funcionamento da aplica????o"
			apt-get install -y git awscli
			#apt-get install -y libhdf5-dev libc-ares-dev libeigen3-dev gcc gfortran libgfortran5 \
            #              libatlas3-base libatlas-base-dev libopenblas-dev libopenblas-base libblas-dev \
            #              liblapack-dev cython3 libatlas-base-dev openmpi-bin libopenmpi-dev python3-dev
			

			 apt-get install ffmpeg libsm6 libxext6  -y
			apt-get install libgl1

			cd /opt

			if [ ! -d "/opt/lookStorestech-peopledetect" ] 
			then
				echo "# Clonando o projeto"
				git clone https://github.com/jeduoliveira/lookStorestech-peopledetect.git
			fi
			
			cd /opt/lookStorestech-peopledetect

			if [ ! -d "./.venv" ] 
			then
				 pip3 install pip --upgrade
				 python3 -m pip install virtualenv
				 python3 -m virtualenv .venv

			fi
			
			pwd
			. .venv/bin/activate

			
			
			pip3 install keras_applications==1.0.8 --no-deps
			pip3 install keras_preprocessing==1.1.0 --no-deps
			pip3 install numpy==1.19.5
			pip3 install h5py==3.1.0
			pip3 install pybind11
			pip3 install -U six wheel mock
			
			curl -L https://lookstoretech-frontend.s3.amazonaws.com/yolov4.weights -o ./data/yolov4.weights
			curl -L https://lookstoretech-frontend.s3.amazonaws.com/tensorflow-2.7.0-cp37-none-linux_aarch64.whl -o tensorflow-2.7.0-cp37-none-linux_aarch64.whl 
			chmod +x tensorflow-2.7.0-cp37-none-linux_aarch64.whl
			pip3 install tensorflow-2.7.0-cp37-none-linux_aarch64.whl
			rm -f tensorflow-2.7.0-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

			pip3 install -r requirements.txt 
			python3 save_model.py --model yolov4 

			
			apt-get -y autoremove

			rpi-upgrade


			
		;;
        *)
			echo "Error: ${lsb_dist} n??o suportado"
			exit 1  
		;;
    esac
    
	exit 1
}

do_install


#apt update -y
#apt update -y
#apt full-upgrade -y
#apt install python3-dev git awscli python3 python3-pip libffi-dev libssl-dev -y
#apt install python3-dev -y
#apt install -y python3 python3-pip
#
    #git clone https://github.com/jeduoliveira/lookStorestech-peopledetect.gi 
#
    #cd lookStorestech-peopledetect
#
    #baixar o arquivo yolov4.weights em  https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT e salvar em ./data/
#
    #python3 -m venv .venv 
    #source .venv/bin/activate
    #pip3 install -r requirements.txt
#
    #python3 save_model.py --model yolov4 
    #python3 main.py --video 0  --model yolov4  
    #
    #