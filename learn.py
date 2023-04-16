#! /usr/bin/env python

from __future__ import print_function

import argparse
import os
import os.path
import shutil
import sys

from lab.calls.call import Call
from lab.environments import  LocalEnvironment

sys.path.append(f'{os.path.dirname(__file__)}/training')
import training
from good_operator_experiment import run_step_good_operators
from partial_grounding_rules import run_step_partial_grounding_rules
from partial_grounding_aleph import run_step_partial_grounding_aleph
from gnn_training import run_step_gnn_learning
from generate_gnn_data import run_step_generate_gnn_data

from downward import suites


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain", help="path to domain file")
    parser.add_argument("problem", nargs="+", help="path to problem file")
    parser.add_argument("--path", default='./data', help="path to store results")
    parser.add_argument("--cpus", type=int, default=1, help="number of cpus available")
    parser.add_argument("--total_time_limit", default=30, type=int, help="time limit")
    parser.add_argument("--total_memory_limit", default=7*1024, help="memory limit")

    return parser.parse_args()


def main():
    args = parse_args()

    ROOT = os.path.dirname(os.path.abspath(__file__))

    TRAINING_DIR=args.path

    REPO_GOOD_OPERATORS = f"{ROOT}/fd-symbolic"
    REPO_LEARNING = f"{ROOT}/learning"
    BENCHMARKS_DIR = f"{TRAINING_DIR}/instances-training"
    INSTANCES_SMAC = f"{TRAINING_DIR}/instances-smac"

    REPO_GNN_LEARNING = f"{ROOT}/gnn-learning"
    GNN_DATA_DIR = f"{TRAINING_DIR}/gnn-data"
    GNN_LEARNING_DIR = f"{TRAINING_DIR}/gnn-learning"

    if os.path.exists(TRAINING_DIR):
        shutil.rmtree(TRAINING_DIR)
    os.mkdir(TRAINING_DIR)

    # Copy all input benchmarks to the directory
    os.mkdir(BENCHMARKS_DIR)
    shutil.copy(args.domain, BENCHMARKS_DIR)

    # os.mkdir(INSTANCES_SMAC)
    # shutil.copy(args.domain, INSTANCES_SMAC)

    for problem in args.problem:
        # TODO Split instances in some way and only put some on instances smac
        shutil.copy(problem, BENCHMARKS_DIR)
        # shutil.copy(problem, INSTANCES_SMAC)
    
    ENV = LocalEnvironment(processes=args.cpus)
    SUITE_TRAINING = suites.build_suite(TRAINING_DIR, ['instances-training'])

    run_step_good_operators(f'{TRAINING_DIR}/good-operators-unit', REPO_GOOD_OPERATORS, ['--search', "sbd(store_operators_in_optimal_plan=true, cost_type=1)"], ENV, SUITE_TRAINING, fetch_everything=True,)

    # from training.gnn_training import run_step_gnn_learning
    # from training.generate_gnn_data import run_step_generate_gnn_data
    # run_generate_graph_objects:
    # TODO
    data_folders = []
    good_operators_data = f'{TRAINING_DIR}/good-operators-unit'
    gnn_data_good_ops = f'{GNN_DATA_DIR}/good-operators-unit'
    gnn_model_data_good_ops = f'{GNN_LEARNING_DIR}/good-operators-unit'
    # x2 = f'{TRAINING_DIR}/lama'
    # y2 = f'{GNN_DATA_DIR}/lama'
    data_folders.append((good_operators_data, gnn_data_good_ops, "good_operators"))
    # data_folders.append((x2, y2))

    for problems_dir, output_dir, good_action_file_name in data_folders:
        run_step_generate_gnn_data(
            REPO_GNN_LEARNING=REPO_GNN_LEARNING,
            PROBLEMS_DIR=problems_dir,
            OUTPUT_DIR=output_dir,
            good_actions_file_name=good_action_file_name,
        )
    
    run_step_gnn_learning(
        REPO_LEARNING=REPO_GNN_LEARNING,
        problems_dir=gnn_data_good_ops,
        output_dir=gnn_model_data_good_ops,
        time_limit=300,
        memory_limit=4 *1024 *1024
    )


if __name__ == "__main__":
    main()
