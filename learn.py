#! /usr/bin/env python

from __future__ import print_function

import argparse
import os
import os.path
import shutil
import sys

from lab.calls.call import Call
from lab.environments import  LocalEnvironment

debug=False

sys.path.append(f'{os.path.dirname(__file__)}/training')
import training
if debug:
    from training.gnn_training import run_step_gnn_learning
    from training.generate_gnn_data import run_step_generate_gnn_data
    from training.run_experiment import RunExperiment
    from training.utils import (
        select_instances_with_properties,
        save_model,
    )
    from training.instance_set import InstanceSet, select_instances_from_runs
    from training.optimize_smac import run_smac

else:
    from gnn_training import run_step_gnn_learning
    from generate_gnn_data import run_step_generate_gnn_data
    from run_experiment import RunExperiment
    from utils import (
        save_model,
    )
    from instance_set import InstanceSet, select_instances_from_runs
    from optimize_smac import run_smac




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
    BENCHMARKS_DIR = f"{TRAINING_DIR}/instances"
    INSTANCES_SMAC = f"{TRAINING_DIR}/instances-smac"

    REPO_GNN_LEARNING = f"{ROOT}/gnn-learning"
    GNN_DATA_DIR = f"{TRAINING_DIR}/gnn-data"
    GNN_LEARNING_DIR = f"{TRAINING_DIR}/gnn-learning"


    if os.path.exists(TRAINING_DIR):
        shutil.rmtree(TRAINING_DIR)
    os.mkdir(TRAINING_DIR)

    # Copy all input benchmarks to the directory
    if os.path.isdir(args.domain): # If the first argument is a folder instead of a domain file
        shutil.copytree(args.domain, BENCHMARKS_DIR)
        args.domain += "/domain.pddl"
    else:
        os.mkdir(BENCHMARKS_DIR)
        shutil.copy(args.domain, BENCHMARKS_DIR)
        for problem in args.problem:
            shutil.copy(problem, BENCHMARKS_DIR)

    
    ENV = LocalEnvironment(processes=args.cpus)
    SUITE_ALL = suites.build_suite(TRAINING_DIR, ['instances'])

    # Overall time limit is 10s and 1G # TODO: Set suitable time and memory limit
    RUN = RunExperiment (10, 1000)

    RUN.run_planner(f'{TRAINING_DIR}/runs-lama', REPO_GOOD_OPERATORS, [], ENV, SUITE_ALL, driver_options = ["--alias", "lama-first"])

    instances_manager = InstanceSet(f'{TRAINING_DIR}/runs-lama')

    # We run the good operators tool only on instances solved by lama in less than 30 seconds
    instances_to_run_good_operators = instances_manager.select_instances([lambda i, p : p['search_time'] < 30])

    SUITE_GOOD_OPERATORS = suites.build_suite(TRAINING_DIR, [f'instances:{name}.pddl' for name in instances_to_run_good_operators])
    RUN.run_good_operators(f'{TRAINING_DIR}/good-operators-unit', REPO_GOOD_OPERATORS, ['--search', "sbd(store_operators_in_optimal_plan=true, cost_type=1)"], ENV, SUITE_GOOD_OPERATORS)
    instances_manager.add_training_data(f'{TRAINING_DIR}/good-operators-unit')

    has_action_cost = len(select_instances_from_runs(f'{TRAINING_DIR}/good-operators-unit', lambda p : p['use_metric'])) > 0
    if has_action_cost:
        RUN.run_good_operators(f'{TRAINING_DIR}/good-operators-cost', REPO_GOOD_OPERATORS, ['--search', "sbd(store_operators_in_optimal_plan=true)"], ENV, SUITE_GOOD_OPERATORS)
        instances_manager.add_training_data(f'{TRAINING_DIR}/good-operators-cost')

    TRAINING_INSTANCES = instances_manager.split_training_instances()

    
    # TODO
    data_folders = []
    good_operators_data = f'{TRAINING_DIR}/good-operators-unit'
    gnn_data_good_ops = f'{GNN_DATA_DIR}/good-operators-unit'
    gnn_model_data_good_ops = f'{GNN_LEARNING_DIR}/good-operators-unit'

    lama_operators_data = f'{TRAINING_DIR}/runs-lama'
    gnn_data_lama_ops = f'{GNN_DATA_DIR}/runs-lama'
    gnn_model_data_lama_ops = f'{GNN_LEARNING_DIR}/runs-lama'
     

    # x2 = f'{TRAINING_DIR}/lama'
    # y2 = f'{GNN_DATA_DIR}/lama'
    data_folders.append((good_operators_data, gnn_data_good_ops, "good_operators"))
    data_folders.append((lama_operators_data, gnn_data_lama_ops, "sas_plan"))
    # data_folders.append((x2, y2))

    for problems_dir, output_dir, good_action_file_name in data_folders:
        run_step_generate_gnn_data(
            REPO_GNN_LEARNING=REPO_GNN_LEARNING,
            PROBLEMS_DIR=problems_dir,
            OUTPUT_DIR=output_dir,
            good_actions_file_name=good_action_file_name,
        )


        
    
    # run_step_gnn_learning(
    #     REPO_LEARNING=REPO_GNN_LEARNING,
    #     problems_dir=gnn_data_good_ops,
    #     output_dir=gnn_model_data_good_ops,
    #     training_dir=TRAINING_DIR,
    #     time_limit=300,
    #     memory_limit=4 *1024 *1024
    # )

    # run_step_gnn_learning(
    #     REPO_LEARNING=REPO_GNN_LEARNING,
    #     problems_dir=gnn_data_good_ops,
    #     output_dir=gnn_model_data_lama_ops,
    #     training_dir=TRAINING_DIR,
    #     time_limit=300,
    #     memory_limit=4 *1024 *1024
    # )



    

    SMAC_INSTANCES = instances_manager.get_smac_instances(['translator_operators', 'translator_facts', 'translator_variables'])
  

    run_smac( ROOT, f'{TRAINING_DIR}', f'{TRAINING_DIR}/smac1', args.domain, BENCHMARKS_DIR, SMAC_INSTANCES, walltime_limit=100, n_trials=100, n_workers=1)

if __name__ == "__main__":
    main()
