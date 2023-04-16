#! /usr/bin/env python

from __future__ import print_function

import argparse
import os.path
import subprocess
import sys
import shutil


ROOT = os.path.dirname(os.path.abspath(__file__))

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain_knowledge", help="path to domain knowledge file")
    parser.add_argument("domain", help="path to domain file")
    parser.add_argument("problem", help="path to problem file")
    parser.add_argument("plan", help="path to output plan file")
    return parser.parse_args()


def main():
    args = parse_args()
    # best_model_path = os.join.path(args.dk, "best_model/model.pt")

    ROOT = os.path.dirname(os.path.abspath(__file__))
    REPO_GNN_LEARNING = f"{ROOT}/gnn-learning"
    SCORPION_PATH = f"{ROOT}/scorpion"

    DK_DIR_FILE = args.domain_knowledge
    DOMAIN = args.domain
    PROBLEM = args.problem
    PLAN_OUT = args.plan
    
    DK_DIR = f'{ROOT}/DK'

    shutil.unpack_archive(DK_DIR_FILE, DK_DIR ,'zip')
    
    model_path = os.path.join(DK_DIR, "model.pt")
    preprocessor_settings = os.path.join(DK_DIR, "preprocessor_settings.txt")

    # Read preprocessor settings as string
    with open(preprocessor_settings, "r") as f:
        preprocessor_settings = f.read()

    subprocess.check_call([f'{SCORPION_PATH}/fast-downward.py', '--alias', 'lama', '--transform-task-options', preprocessor_settings, '--transform-task', f'{REPO_GNN_LEARNING}/src/preprocessor.command', DOMAIN, PROBLEM])



if __name__ == "__main__":
    main()
