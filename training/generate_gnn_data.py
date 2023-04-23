import os 
import sys
from lab.calls.call import Call

def run_step_generate_gnn_data(REPO_GNN_LEARNING, PROBLEMS_DIR, OUTPUT_DIR, time_limit=300, memory_limit = 4*1024*1024):
    Call([sys.executable, f'{REPO_GNN_LEARNING}/src/graph_data_generation.py', PROBLEMS_DIR, OUTPUT_DIR], "generate-graphs-gnn" ,time_limit=time_limit, memory_limit=memory_limit).wait()

