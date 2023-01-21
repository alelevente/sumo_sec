FROM ubuntu

RUN apt update
RUN apt install -y sumo sumo-tools
RUN apt install -y pip
RUN pip install numpy
RUN pip install pandas
RUN pip install lxml
ENV SUMO_HOME=/usr/share/sumo

RUN mkdir MoSTScenario
COPY build_inputs/MoSTScenario/. MoSTScenario/.
COPY 03_src/01_simulate.py /simulate.py
