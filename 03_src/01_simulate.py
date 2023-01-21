'''
    This script conducts the measurments of the MoST scenario according to the
    simulation configuration file.
'''

import os, sys
import numpy as np
#import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import pandas as pd

import json
import argparse

#from tqdm import trange


#SUMO imports:
SUMO_HOME = os.environ["SUMO_HOME"] #locating the simulator
sys.path.append(SUMO_HOME+"/tools")
import traci
import sumolib

#constant definitions:
MOST_ROOT= "MoSTScenario/scenario/"
PARKING_AREA_DEFINITIONS = "%sin/add/most.parking.add.xml"%MOST_ROOT

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "dockerizable TraCI measurement tool")
    parser.add_argument("--name", help="name of the simulation", type=str)
    parser.add_argument("--threads", help="no. of execution threads", type=int)
    CONFIG = parser.parse_args()
    OUTPUT_DIR = "/measurement/%s/"%CONFIG.name
    try:
        os.mkdir(OUTPUT_DIR)
    except:
        pass

    
    #loading the configuration files:
    parking_df = pd.read_xml(PARKING_AREA_DEFINITIONS)
    parking_ids = parking_df["id"].values.tolist()
    
    #Starting TraCI:
    sumoBinary = "sumo"
    sumoCmd = [sumoBinary, "-c", "%smost.sumocfg"%MOST_ROOT, "--threads", str(CONFIG.threads),
               "--device.rerouting.threads", str(CONFIG.threads),
               "--random", "--random-depart-offset", "900", "--verbose", "false"]
    sumoCmd += ["--net-file", "%sin/most.net.xml"%MOST_ROOT]
    sumoCmd += ["--additional-files", "%sin/add/most.poly.xml,%sin/add/most.busstops.add.xml,%sin/add/most.trainstops.add.xml,%sin/add/most.parking.allvisible.add.xml,%sin/add/basic.vType.xml,%sin/route/most.pedestrian.rou.xml,%sin/route/most.commercial.rou.xml,%sin/route/most.special.rou.xml,%sin/route/most.highway.flows.xml"%(MOST_ROOT,
    MOST_ROOT,MOST_ROOT,MOST_ROOT,MOST_ROOT,MOST_ROOT,MOST_ROOT,MOST_ROOT,MOST_ROOT)]

    traci.start(sumoCmd)
        
    #measurement:
    id_list = []
    timestamps = []
    counts = []
    sims = []

    edge_to_idx_map = {}
    idx_to_edge_map = {}
    vehicle_to_idx_map = {}
    idx_to_vehicle_map = {}

    edges = traci.edge.getIDList()

    for i,e in enumerate(edges):
        edge_to_idx_map[e] = i
        idx_to_edge_map[i] = e
        
    #saving the encodings:
    config = {"edge_to_idx_map": edge_to_idx_map,
          "idx_to_edge_map": idx_to_edge_map}
    with open("%sedge_maps.json"%OUTPUT_DIR, "w") as outfile:
        json.dump(config, outfile)

    #to save memory, we will write results instantly into files:
    files = open("%s_vehicle_positions.csv"%OUTPUT_DIR, "w")
    files.write("ids,timestamp,position\n")

    #main simulation loop:
    for step in range(14400, 50400, 1):
        step += 10
        if step%60 == 0:
            #asking for parking lots:
            for p in parking_ids:
                count = traci.parkingarea.getVehicleCount(str(p))
                id_list.append(p)
                timestamps.append(step)
                counts.append(count)
        #asking for vehicles:
        vehicles = traci.vehicle.getIDList()
        for veh in vehicles:
            if not(veh in vehicle_to_idx_map):
                vehicle_to_idx_map[veh] = len(vehicle_to_idx_map)
                idx_to_vehicle_map[len(idx_to_vehicle_map)] = veh
            veh_idx = vehicle_to_idx_map[veh]
            pos = edge_to_idx_map[traci.vehicle.getRoadID(str(veh))]
            files.write("%d,%d,%d\n"%(veh_idx, step, pos))
        traci.simulationStep(step)

    traci.close()

    #saving results:
    files.close()
        
    vehicles = {"vehicle_to_idx_map": vehicle_to_idx_map,
          "idx_to_vehicle_map": idx_to_vehicle_map}
    with open("%svehicle_maps.json"%OUTPUT_DIR, "w") as outfile:
        json.dump(vehicles, outfile)
    occup_df = pd.DataFrame({
        "ids": id_list,
        "timestamp": timestamps,
        "counts": counts,
    })
    
    occup_df.to_csv("%soccupancies.csv"%OUTPUT_DIR, index=False)
