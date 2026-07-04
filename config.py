import os

#############################

ROOT="."

#############################

DATASET=os.path.join(ROOT,"datasets")

VSUMM=os.path.join(DATASET,"VSUMM")

TVSUM=os.path.join(DATASET,"TVSum")

SUMME=os.path.join(DATASET,"SumMe")

ACCIDENT=os.path.join(DATASET,"Accident")

#############################

IMG_SIZE=224

NUM_FRAMES=16

FPS=1

#############################

BATCH_SIZE=4

LR=1e-4

EPOCHS=30

#############################

DEVICE="cuda"

#############################

CHECKPOINT="checkpoints"

RESULT="results"

os.makedirs(CHECKPOINT,exist_ok=True)

os.makedirs(RESULT,exist_ok=True)