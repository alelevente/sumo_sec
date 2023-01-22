"""
    Conducts measurement in multiple containers
"""
import os
from multiprocessing.pool import ThreadPool
import argparse

THREADS = 16


def measurement_runner(meas_name):
    os.system('docker run -v "$(pwd)"/dockeroutput:/measurement:z simulator python3 simulate.py --name %s --threads 1'%meas_name)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog = "Dockerized SUMO runner script")
    parser.add_argument('n_measurements')
    args = parser.parse_args()

    n_meas = int(args.n_measurements)

    pool = ThreadPool(processes=THREADS)
    
    items = ["day_%d"%i for i in range(n_meas)]
    pool.map(measurement_runner, items)
