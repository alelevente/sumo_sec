# To be able to use the package manager:
FROM ubuntu
RUN apt update

# Installing the latest SUMO:
RUN apt install -y software-properties-common
RUN add-apt-repository -y ppa:sumo/stable
RUN apt update
RUN apt install -y sumo sumo-tools
ENV SUMO_HOME=/usr/share/sumo

# Installing Python and required packages of the measurement script:
RUN apt install -y pip
RUN pip install numpy
RUN pip install pandas
RUN pip install lxml

# Copying the scenario and the measurement script:
RUN mkdir MoSTScenario
COPY build_inputs/MoSTScenario/. MoSTScenario/.
COPY 03_src/01_simulate.py /simulate.py
