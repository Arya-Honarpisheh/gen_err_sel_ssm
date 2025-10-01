#!/bin/bash

# Define the variables for the experiment parameters
DEVID=0                             # GPU device ID
USEDELTA=True                       # Use delta in SSM
FIXSA=False                         # Fix s_A in the A matrix first entry
SAVE=True                           # Save the model
MODEL_PATH=''                       # Path to pre-trained model (if skipping training)
SEED=22                             # Random seed

export PYTHONHASHSEED=$SEED

# Choose between 'stability_margin' or 'length_independence'
EXPERIMENT="length_independence"  # Experiment type

# Choose sequence lenght T and stability margin s_A values based on the experiment 
if [ "$EXPERIMENT" == "stability_margin_T" ]; then
    T_VALUES="300 350 400 450 500 550 600"
    # T_VALUES="50 100 150 200 250 300 350"
    T_VALUES="100 200 300 400 500 600 700"
    S_A_VALUES="0.1"
elif [ "$EXPERIMENT" == "length_independence" ]; then
    T_VALUES="25 50 75 100 125 150 175 200 225 250 275 300 325 350 375 400"
    T_VALUES="1800 1900"
    S_A_VALUES="0.0"
fi

# Choose between 'imdb' and 'majority'
DATASET="imdb" # Dataset to use

if [ "$DATASET" == "imdb" ]; then
    NUM_TRAIN=25000                                         # Number of training samples
    NUM_VALID=0                                             # Number of validation samples
    NUM_TEST=25000                                          # Number of test samples
    BALANCED=False                                          # Balanced dataset flag
    N=4                                                     # Number of states per channel in SSM
    D=16                                                    # Number of channels in SSM
    NUM_EPOCHS=30                                           # Number of epochs to train
    BATCH_SIZE=256                                          # Batch size for training
    LEARNING_RATE=1e-2                                      # Learning rate for optimizer
    WEIGHT_DECAY=1e-5                                       # Weight decay for optimizer
    TVAR=0                                                  # Time variability
elif [ "$DATASET" == "majority" ]; then
    NUM_TRAIN=1000
    NUM_VALID=0
    NUM_TEST=1000
    BALANCED=False
    N=4
    D=4
    NUM_EPOCHS=10
    BATCH_SIZE=64
    LEARNING_RATE=1e-2
    WEIGHT_DECAY=1e-5
    TVAR=0
elif [ "$DATASET" == "listops" ]; then
    NUM_TRAIN=1000
    NUM_VALID=0
    NUM_TEST=1000
    BALANCED=False
    N=4
    D=16
    NUM_EPOCHS=10
    BATCH_SIZE=64
    LEARNING_RATE=1e-2
    WEIGHT_DECAY=1e-5
    TVAR=5
fi

# Run the Python script with the variables passed as arguments
python main.py \
    --devID $DEVID \
    --experiment $EXPERIMENT \
    --dataset $DATASET \
    --T_values $T_VALUES \
    --T_var $TVAR \
    --s_A_values $S_A_VALUES \
    --N $N \
    --d $D \
    --num_train $NUM_TRAIN \
    --num_valid $NUM_VALID \
    --num_test $NUM_TEST \
    --balanced $BALANCED \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --use_delta $USEDELTA \
    --fix_sA $FIXSA \
    --save_results $SAVE \
    --seed $SEED