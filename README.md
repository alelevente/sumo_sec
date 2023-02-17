# SUMO Simulations for Secure AV Learning

This repository contains codes which serves as a basis for our article entitled as `SUMO Simulations for Secure Machine Learning in
Communicating Autonomous Vehicles` submitted to the [SUMO User Conference 2023](https://www.eclipse.org/sumo/conference/#agenda).

## Prerequisites

Besides installing the used python packages, one shall also make the following steps to run the codes inside this repository:

1. [Install Eclipse SUMO](https://sumo.dlr.de/docs/Installing/index.html).
2. [Download the MoST scenario](https://github.com/lcodeca/MoSTScenario/tree/master/scenario)
3. Place the MoST scenario into the `build_inputs` folder. Replace `build_inputs/MoSTScenario/scenario/most.sumocfg` by the file inside our repository.
4. Create a folder named `dockeroutput` into the root of this repository.
5. Run the command: `docker build -t simulator .` in the root of this repository.

## Source codes:
In folder `03_src` one can find our source codes. Please note, `01_data_collector.ipynb` notebook only serves development reasons.

On branch `server_based`, one may find codes that allows training on `Flask`-based worker servers.