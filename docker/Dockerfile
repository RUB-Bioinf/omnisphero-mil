# the provided base image:
FROM 		nvidia/cuda:11.0-base
# Maintainer
MAINTAINER	sven.kreienbrock@ruhr-uni-bochum.de

RUN apt-key del "7fa2af80" \
&& export this_distro="$(cat /etc/os-release | grep '^ID=' | awk -F'=' '{print $2}')" \
&& export this_version="$(cat /etc/os-release | grep '^VERSION_ID=' | awk -F'=' '{print $2}' | sed 's/[^0-9]*//g')" \
&& apt-key adv --fetch-keys "https://developer.download.nvidia.com/compute/cuda/repos/${this_distro}${this_version}/x86_64/3bf863cc.pub"

# default system update & upgrade
RUN	apt-get -qq -y update && apt-get -qq -y upgrade

# install some necessary base packages
RUN	apt-get -y -qq install vim wget curl apt-transport-https git

# install R
ENV	DEBIAN_FRONTEND=noninteractive
RUN	apt-get -y -qq install r-base r-base-core r-base-dev
RUN	apt-get -y -qq install r-cran-boot r-cran-class r-cran-cluster r-cran-codetools \
	r-cran-foreign r-cran-kernsmooth r-cran-lattice r-cran-mass r-cran-matrix \
	r-cran-mgcv r-cran-nlme r-cran-nnet r-cran-rpart r-cran-spatial \
	r-cran-survival r-doc-html r-recommended 

# manually install anaconda individual edition in silent mode:
ENV	ANACONDA_VER=Miniconda3-py38_4.10.3-Linux-x86_64.sh
RUN	wget https://repo.anaconda.com/miniconda/$ANACONDA_VER
RUN	/bin/bash $ANACONDA_VER -b -u -p /opt/anaconda3
# delete Anaconda-Installer
RUN	rm -v $ANACONDA_VER
# set environments and install prerequisites
ENV	PATH=/opt/anaconda3/bin:$PATH

# setup environment
WORKDIR	/opt
# COPY	py38.condaenv /opt/py38.yml
RUN curl -o /opt/py38.yml http://git.prodi.ruhr-uni-bochum.de/bioinf/public/raw/branch/master/dockerfiles/bioinfbo/py38gpuR/py38.condaenv
RUN	conda init bash 
RUN 	conda env create -f py38.yml

# clean up the system installation
RUN	apt-get -y autoremove && apt-get clean

# Install R-Packages
RUN	R -e "install.packages('drc', repos='http://cran.rstudio.com/')" \
	R -e "install.packages('Rserve', repos='http://cran.rstudio.com/')"

# Running the Rserve in the background
RUN R -e "library(Rserve);Rserve();" &

# add the necessary user & group (security)
# default uid/gid 1000: ubuntu default user
# feel free to change this to your own username and uid/gid!
RUN	groupadd -g 1000 biox
RUN	useradd -m -u 1000 -g biox localuser
RUN	mkdir /opt/notebooks && chown localuser.biox /opt/notebooks
# set the user that later runs the software:
USER	localuser
RUN	conda init && echo "conda activate py38" >>/home/localuser/.bashrc
WORKDIR	/home/localuser
