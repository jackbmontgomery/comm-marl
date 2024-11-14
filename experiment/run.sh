#!/bin/bash

# Define variables
environments=("PredatorPrey" "TrafficJunction")
with_comm_options=(1 0)
n_steps=300000

python param_share/main.py --env "PredatorPrey" --with_comm 0 --n_steps "$n_steps"
python param_share/main.py --env "PredatorPrey" --with_comm 0 --n_steps "$n_steps"
python param_share/main.py --env "TrafficJunction" --with_comm 0 --n_steps "$n_steps"
python param_share/main.py --env "TrafficJunction" --with_comm 0 --n_steps "$n_steps"

# NO PARAM SHARING
for env in "${environments[@]}"; do
    for with_comm in "${with_comm_options[@]}"; do
        echo "Running with env=$env and with_comm=$with_comm for no_param_share"
        python no_param_share/main.py --env "$env" --with_comm "$with_comm" --n_steps "$n_steps"
    done
done

for i in {1..3}; do
    echo "Iteration $i of 3"

    # NO PARAM SHARING
    for env in "${environments[@]}"; do
        for with_comm in "${with_comm_options[@]}"; do
            echo "Running with env=$env and with_comm=$with_comm for no_param_share"
            python no_param_share/main.py --env "$env" --with_comm "$with_comm" --n_steps "$n_steps"
        done
    done

    # PARAM SHARING
    for env in "${environments[@]}"; do
        for with_comm in "${with_comm_options[@]}"; do
            echo "Running with env=$env and with_comm=$with_comm for param_share"
            python param_share/main.py --env "$env" --with_comm "$with_comm" --n_steps "$n_steps"
        done
    done

done