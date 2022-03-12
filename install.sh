
#!/bin/sh
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
	echo "# Executing install script"
    user="$(id -un 2>/dev/null || true)"

	sh_c='sh -c'
	if [ "$user" != 'root' ]; then
			cat >&2 <<-'EOF'
			Error: this installer needs the ability to run commands as root.
			We are unable to find either "sudo" or "su" available to make this happen.
			EOF
			exit 1  
	fi

    lsb_dist=$( get_distribution )
	lsb_dist="$(echo "$lsb_dist" | tr '[:upper:]' '[:lower:]')"

    case "$lsb_dist" in
        debian|raspbian)
			echo "# remove python"
			apt-get -y remove python3 >/dev/null
			apt-get autoremove -y >/dev/null

			echo "# enable ssh"
			systemctl enable ssh  >/dev/null
			systemctl start ssh  >/dev/null

			echo "# update"
			apt-get update  >/dev/null
			apt-get -y upgrade  >/dev/null

			echo "# install packages"
			apt-get -y install git awscli \ 
			build-essential tk-dev libncurses5-dev libncursesw5-dev libreadline6-dev \ 
			libdb5.3-dev libgdbm-dev libsqlite3-dev libssl-dev libbz2-dev libexpat1-dev liblzma-dev zlib1g-dev libffi-dev

			echo "# install python 3.8"
			cd /opt
			if [ ! -d "/opt/Python-3.8.12" ] 
			then
				wget https://www.python.org/ftp/python/3.8.12/Python-3.8.12.tgz  >/dev/null
				tar xzf Python-3.8.12.tgz  >/dev/null
			fi
			cd Python-3.8.12
			./configure --enable-optimizations
			make
		    make altinstall
			cd ..
			rm -r Python-3.8.12
			rm Python-3.8.12.tgz
			. ~/.bashrc
			update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1
			python -V
			
			echo "# install pip"
			cd ..
			curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
			python3.8 get-pip.py

			if [ ! -d "/opt/lookStorestech-peopledetect" ] 
			then
				 git clone https://github.com/jeduoliveira/lookStorestech-peopledetect.git
			fi

			cd /opt/lookStorestech-peopledetect

			if [ ! -d "./.venv" ] 
			then
				 python3 -m venv .venv
			fi
			
			pwd
			. .venv/bin/activate
			pip3 install -r requirements.txt


			
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