"""SN+CNN 500 samples Cifar-10
"""

import os
import math
import time
import argparse

import numpy as np

from multiprocessing import Process

os.environ['MKL_THREADING_LAYER'] = 'GNU' # Fix a bug : mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1 library.
        #Try to import numpy first or set the threading layer accordingly. Set MKL_SERVICE_FORCE_INTEL to force it.

PROCESS_BATCH_SIZE = 2

mlflow_exp_name = "\"SN+CNN 100 samples Cifar-10\""

PYTHON = ''
RUN_FILE = "parametricSN/main.py"
OPTIM = "sgd"
LR = 0.1
LRS = 0.1
LRO = 0.1
LRMAX = 0.1
DF = 25
THREE_PHASE = 1
SEED = int(time.time() * np.random.rand(1))
LEARNABLE = 1
EPOCHS = 3000
INIT = "Tight-Frame"
RUNS_PER_SEED = 10
SCHEDULER = "OneCycleLR"
TRAIN_SAMPLE_NUM = 100
TRAIN_BATCH_SIZE = 128
AUGMENT = "autoaugment"
MODEL = "cnn"
PHASE_ENDS = " ".join(["100","200"])
MODEL_LOSS = 'cross-entropy'
SCATT_LRMAX = 0.2
SCATT_DF = 25
SCATT_THREE_PHASE = 1


def runCommand(cmd):
    print("[Running] {}".format(cmd))
    os.system(cmd)

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-dr", type=str)
    parser.add_argument("--data-folder", "-df", type=str)
    parser.add_argument("--python", "-p", type=str)

    return parser.parse_args()

if __name__ == '__main__':
    args = cli()

    if args.data_root != None and args.data_folder != None:
        DATA_ARG = "-ddr {} -ddf {}".format(args.data_root,args.data_folder)
    else:
        DATA_ARG = ""

    if args.python != None:
        PYTHON = args.python

    commands = []

    for SEED in [207715039, 491659600,737523103,493572006,827192296,877498678,1103100946,1210393663,1277404878,1377264326]:
        for aa in [(1,"Tight-Frame")]:
            LEARNABLE, INIT = aa

            args1 = "-oname {} -olr {} -gseed {} -sl {} -me {} -omaxlr {} -odivf {} -sip {} -dtsn {}".format(
                OPTIM,LR,SEED,LEARNABLE,EPOCHS,LRMAX,DF,INIT,TRAIN_SAMPLE_NUM
            )

            args2 = "-os {} -daug {} -en {} -dtbs {} -mname {} -ope {}".format(
                SCHEDULER,AUGMENT,mlflow_exp_name,TRAIN_BATCH_SIZE,MODEL,PHASE_ENDS
            )

            args3 = "-smaxlr {} -sdivf {} -stp {} -mloss {}".format(
                SCATT_LRMAX,SCATT_DF,SCATT_THREE_PHASE,MODEL_LOSS
            )

            command = "{} {} run-train {} {} {} {}".format(
                PYTHON,RUN_FILE,args1,args2,args3,DATA_ARG)

            commands.append(command)

    

    for cmd in commands:
        print(cmd)

    processes = [Process(target=runCommand,args=(commands[i],)) for i,cmd in enumerate(commands)]
    processBatches = [processes[i*PROCESS_BATCH_SIZE:(i+1)*PROCESS_BATCH_SIZE] for i in range(math.ceil(len(processes)/PROCESS_BATCH_SIZE))]

    for i,batch in enumerate(processBatches):
        print("Running process batch {}".format(i))
        startTime = time.time()

        for process in batch:
            process.start()
            time.sleep(5)

        for process in batch:
            process.join()

        print("\n\nRunning Took {} seconds".format(time.time() - startTime))
        time.sleep(1)






