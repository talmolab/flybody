#!/bin/bash

# Run the script for each configuration
python offline_learner.py --config-name="train_config_mouse_reach_offline_muscle_simple"
python offline_learner.py --config-name="train_config_mouse_reach_offline_muscle"
python offline_learner.py --config-name="train_config_mouse_reach_offline_position"
python offline_learner.py --config-name="train_config_mouse_reach_offline_torque"
