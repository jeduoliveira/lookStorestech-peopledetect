
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
	#echo $user
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
        debian|raspbian)
			sudo cd /opt

			echo "# Habilitando  ssh"
			sudo systemctl enable ssh  
			sudo systemctl start ssh  

			echo "# Realizando update e upgrade do SO"
			sudo apt-get update  
			sudo apt-get -y full-upgrade 

			echo "# Instalando pacotes necessarios para o funcionamento da aplicação"
			sudo apt-get install -y git awscli
			sudo apt-get install -y libhdf5-dev libc-ares-dev libeigen3-dev gcc gfortran libgfortran5 libatlas3-base libatlas-base-dev libopenblas-dev libopenblas-base libblas-dev liblapack-dev cython3 libatlas-base-dev openmpi-bin libopenmpi-dev python3-dev
 
			if [ ! -d "/opt/lookStorestech-peopledetect" ] 
			then
				echo "# Clonando o projeto"
				sudo git clone https://github.com/jeduoliveira/lookStorestech-peopledetect.git
			fi
			
			sudo cd /opt/lookStorestech-peopledetect

			if [ ! -d "./.venv" ] 
			then
				 sudo python3 -m pip install virtualenv
				 sudo python3 -m virtualenv .venv
			fi
			
			pwd
			sudo . .venv/bin/activate
			
			sudo pip install -U wheel mock six
			echo "# Download tensorflow wheels"
			sudo curl -L https://github.com/PINTO0309/Tensorflow-bin/releases/download/v2.8.0/tensorflow-2.8.0-cp39-none-linux_aarch64.whl -o tensorflow-2.8.0-cp39-none-linux_aarch64.whl
			
			sudo chmod +x tensorflow-2.8.0-cp39-none-linux_aarch64.whl
			sudo pip uninstall tensorflow
			sudo pip install tensorflow-2.8.0-cp39-none-linux_aarch64.whl

			sudo pip3 install -r requirements.txt
			sudo reboot
		;;
        *)
			echo "Error: ${lsb_dist} não suportado"
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