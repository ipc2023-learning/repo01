#!/usr/bin/env python

import sys
import os
import logging
import argparse

from preprocessor.helper_functions import save_to_workspace, copy_file
from preprocessor.preprocess_h2 import run_h2_preprocessor_on_file
from preprocessor.preprocess_GNN import run_gnn_preprocessor

# logger with info display level
logger = logging.getLogger(__name__)
# set info to be visible
logger.setLevel(logging.INFO)

H2_PATH = "workspace/h2.sas"
H2_GNN_PATH = "workspace/h2_gnn.sas"
H2_GNN_H2_PATH = "workspace/h2_gnn_h2.sas"
WORKSPACE_SAS = "workspace/output.sas"
# import dupa

def default_gnn_preprocessor(threshold, retries, model_path):
    # Original step
    input = sys.stdin
    save_to_workspace(input, "original.sas")
    with open(WORKSPACE_SAS, "r") as f:
        workspace_original_sas = f.read()

    # H2 step
    run_h2_preprocessor_on_file(WORKSPACE_SAS)
    copy_file(WORKSPACE_SAS, H2_PATH)
    # Check 
    with open(WORKSPACE_SAS, "r") as f:
        h2_sas = f.read()
    if workspace_original_sas == h2_sas:
        logger.info("ORG -> H2 preprocessor did not change the sas file")

    # GNN STEP
    run_gnn_preprocessor(sas_path=WORKSPACE_SAS,
                         output_dir="workspace",
                         model_path=model_path,
                         threshold=threshold,
                         retries=retries)
    copy_file(WORKSPACE_SAS, H2_GNN_PATH)
    # Check
    with open(WORKSPACE_SAS, "r") as f:
        h2_gnn_sas = f.read()
    if h2_sas == h2_gnn_sas:
        logger.info("H2 -> GNN preprocessor did not change the sas file")

    # Second H2 STEP
    run_h2_preprocessor_on_file(WORKSPACE_SAS)
    copy_file(WORKSPACE_SAS, H2_GNN_H2_PATH)
    # Check
    with open(WORKSPACE_SAS, "r") as f:
        h2_gnn_h2_sas = f.read()
    if h2_gnn_sas == h2_gnn_h2_sas:
        logger.info("GNN -> H2 preprocessor did not change the sas file")


def failed_gnn_preprocessor(failed_count, retries):
    if failed_count == retries or not os.path.exists("workspace/retries"):
        if failed_count == retries:
            print("Retries exceeded, using original sas file after h2 preprocessor")

        else:
            print("No retries available")

        copy_file(H2_PATH, WORKSPACE_SAS)
        raise Exception(
            "Retries exceeded or no retries available, using original sas file after h2 preprocessor,\
                this exception was called to indiciate that to the driver"
            )


    # We have that one
    h2_gnn_path = os.path.join("workspace", "retries", f"h2_gnn{failed_count}.sas")
    # We will create this one
    h2_gnn_h2_path = os.path.join("workspace", "retries", f"h2_gnn{failed_count}_h2.sas")

    # H2 Step
    run_h2_preprocessor_on_file(h2_gnn_path)
    copy_file(WORKSPACE_SAS, h2_gnn_h2_path)

    with open(h2_gnn_path, "r") as h2_gnn_f:
        h2_gnn_sas = h2_gnn_f.read()
    with open(h2_gnn_h2_path, "r") as h2_gnn_h2_f:
        h2_gnn_h2_sas = h2_gnn_h2_f.read()

    if h2_gnn_sas == h2_gnn_h2_sas:
        logger.info("GNN -> H2 preprocessor did not change the sas file")
    


argparser = argparse.ArgumentParser()
argparser.add_argument("--gnn-threshold", help="number of retries for gnn", type=float, required=True)
argparser.add_argument("--gnn-retries", help="number of retries for gnn", type=int, required=True)
argparser.add_argument("--failed", help="0 based index, indicating if we have already failed a plan", default=-1, type=int)
argparser.add_argument("--model-path", help="path to the weights of gnn model")


args = argparser.parse_args()
threshold = args.gnn_threshold
retries = args.gnn_retries
model_path = args.model_path


print("##"*100)
if args.failed >= 0:
    failed_gnn_preprocessor(args.failed, retries)

else:
    print("RUNNING DEFAULT TRANSFORMATION")
    default_gnn_preprocessor(
        threshold=threshold,
        retries=retries,
        model_path=model_path
    )

print("##"*100)