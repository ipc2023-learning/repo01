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
    REPO_GNN_LEARNING = f"{ROOT}/gnn-learning"
    BENCHMARKS_DIR = f"{TRAINING_DIR}/instances-training"
    INSTANCES_SMAC = f"{TRAINING_DIR}/instances-smac"
    REPO_PARTIAL_GROUNDING = f"{ROOT}/fd-partial-grounding"

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

    # https://gitlab.com/atorralba/ipc-2023-gnn/-/issues/2
    # Option first is that we will have 2 dirs that follow the some naming convention: eg for problem1.pddl, we have problem1 and problem1.sas
        # good_operators_dir
        # output_sas_dir
    # Option 2 is that we will have 1 dir that has the same name as the problem and inside it we have the sas file and the good operators file 

    # nonetheless, we will make a separate dir, where each of the problems is a specific dir that stores graph objects (csv), and the good operators file

    # run_generate_graph_objects:    
    # run_step_generate_gnn_data(REPO_GNN_LEARNING, f'{TRAINING_DIR}/good-operators-unit', f'{TRAINING_DIR}/gnn-data')

    # train_dir = some
    # something
       # problem1
              # nodes.csv
              # edges.csv
       # problem 2
             # ...

    # run_step_gnn_learning(
    #     REPO_LEARNING=REPO_LEARNING,
    #     WORKING_DIR=os.path.join(TRAINING_DIR, "gnn-learning"),
    #     domain_file=args.domain,
    #     time_limit=300,
    #     memory_limit=4 *1024 *1024
    # )
    # Only do this if the domain has action cost:
    # run_step_good_operators(f'{TRAINING_DIR}/good-operators', REPO_GOOD_OPERATORS, ['--search', "sbd(store_operators_in_optimal_plan=true)"], ENV, SUITE_TRAINING, fetch_everything=True,)

    #TODO: set time and memory limits
    #TODO: train also without good operators
    run_step_partial_grounding_rules(REPO_LEARNING, f'{TRAINING_DIR}/good-operators-unit', f'{TRAINING_DIR}/partial-grounding-rules', args.domain)
    run_step_partial_grounding_aleph(REPO_LEARNING, f'{TRAINING_DIR}/good-operators-unit', f'{TRAINING_DIR}/partial-grounding-aleph', args.domain)

if __name__ == "__main__":
    main()
