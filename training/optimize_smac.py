from ConfigSpace import Categorical, Float, Configuration, ConfigurationSpace, InCondition
from smac import HyperparameterOptimizationFacade, Scenario

from lab.calls.call import Call
from gnn_training import ModelSetting, PreprocessorSettings, run_step_gnn_learning

import sys
import os
import json
import subprocess
import re
import shutil

from collections import defaultdict


# from functools import partial

# Hardcoded paths that depend on the trraining part. This could be passed by parameter instead
PARTIAL_GROUNDING_RULES_DIR = 'partial-grounding-rules'
PARTIAL_GROUNDING_ALEPH_DIR  = 'partial-grounding-aleph'
SUFFIX_ALEPH_MODELS = '.rules'
PREFIX_SK_MODELS = 'model_'

# Hardcoded paths
INTERMEDIATE_SMAC_MODELS = 'intermediate-smac-models'


class Eval:
    def __init__(self, ROOT, DATA_DIR, WORKING_DIR, domain_file, instances_dir):
        self.DATA_DIR = DATA_DIR
        self.MY_DIR = os.path.dirname(os.path.realpath(__file__))
        self.GNN_REPO_DIR = os.path.join(os.path.join(ROOT, "gnn-learning"))
        self.GNN_DATA_DIR = os.path.join(self.DATA_DIR, "gnn-data")
        self.GNN_LEARNING_DIR = os.path.join(self.DATA_DIR, "gnn-learning")

        self.SCORPION_PATH = os.path.join(ROOT, "scorpion")
        # self.candidate_models=candidate_models

        self.SMAC_MODELS_DIR = os.path.abspath(os.path.join(WORKING_DIR, INTERMEDIATE_SMAC_MODELS))
        if os.path.exists(self.SMAC_MODELS_DIR):
            shutil.rmtree(self.SMAC_MODELS_DIR)
        os.mkdir(self.SMAC_MODELS_DIR)
        self.instances_dir = instances_dir
        self.domain_file = domain_file

        self.regex_total_time = re.compile(rb"INFO\s+Planner time:\s(.+)s", re.MULTILINE)
        self.regex_operators = re.compile(rb"Translator operators:\s(.+)", re.MULTILINE)
        self.regex_plan_cost = re.compile(rb"\[t=.*s, .* KB\] Plan cost:\s(.+)\n", re.MULTILINE)
        self.regex_no_solution = re.compile(rb"\[t=.*KB\] Completely explored state space.*no solution.*", re.MULTILINE)

    def target_function(self, config: Configuration, instance: str, seed: int) -> float:
        model_settings, target_folder = parse_config(config)

        DOMAIN = f'{self.instances_dir}/{self.domain_file}'
        PROBLEM = f'{self.instances_dir}/{instance}.pddl'

        print(f"Running {instance} with {model_settings} and {target_folder}")

        model_path = run_step_gnn_learning(self.GNN_REPO_DIR, model_settings, f'{self.GNN_DATA_DIR}/{target_folder}', f'{self.GNN_LEARNING_DIR}/{target_folder}')

        if model_path is None:
            return 10000000
        
        preprocessor_setting = PreprocessorSettings(
            model_path=model_path,
            gnn_retries=3,
            gnn_threshold=0.5
        ).to_parameter_string()

        command = [sys.executable, f'{self.SCORPION_PATH}/fast-downward.py', '--alias', 'lama', '--transform-task-options', preprocessor_setting, '--transform-task', f'{self.GNN_REPO_DIR}/src/preprocessor.command', DOMAIN, PROBLEM]
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


        try:
            output, error_output = proc.communicate(timeout=300) # Timeout in seconds TODO: set externally

            total_time = self.regex_total_time.search(output)
            num_operators = self.regex_operators.search(output)
            plan_cost = self.regex_plan_cost.search(output)

            if total_time and num_operators and plan_cost:
                total_time = float(total_time.group(1))
                num_operators = float(num_operators.group(1))
                plan_cost = float(plan_cost.group(1))
                print (f"Ran {instance} with model settings {model_settings}: time {total_time}, operators {num_operators}, cost {plan_cost}")
                return num_operators
            elif self.regex_no_solution.search(output):
                print (f"Ran {instance} with model settings {model_settings}: not solved due to partial grounding")
                #print(output.decode())
                return 10000000
            else:
                print (f"WARNING: Ran {instance} with model settings {model_settings}: not solved due to unknown reasons")

                print("Output: ", output.decode())
                if error_output:
                    print("Error Output: ", error_output.decode())
                return 10000000
        except subprocess.CalledProcessError:
            print (f"WARNING: Command failed: {' '.join(command)}")
            print (f"Ran {instance} with model settings {model_settings}: not solved due to crash")
            return 10000000

        except subprocess.TimeoutExpired:
            proc.kill()
            print (f"RRan {instance} with model settings {model_settings}: not solved due to time limit")
            return 10000000

        except:
            print (f"Error: Command failed: {' '.join(command)}")

            print("Output: ", output.decode())
            if error_output:
                print("Error Output: ", error_output.decode())


def parse_config(config):
    config_dict = config.get_dictionary()

    modelSettingsDict = {
        'aggr': config_dict['aggr'],
        'conv_type': config_dict['conv_type'],
        'hidden_size': config_dict['hidden_size'],
        'layers_num': config_dict['layers_num'],
        'lr': config_dict['lr'],
        'optimizer': config_dict['optimizer'],
    }
    target_operators = config_dict['target_folder']

    model_settings = ModelSetting.from_dict(modelSettingsDict)

    return model_settings, target_operators

# Note: default configuration should solve at least 50% of the instances. Pick instances
# with LAMA accordingly. If we run SMAC multiple times, we can use different instances
# set, as well as changing the default configuration each time.
def run_smac(ROOT, DATA_DIR, WORKING_DIR, domain_file, instance_dir, instances_with_features : dict, walltime_limit, n_trials, n_workers):
    DATA_DIR = os.path.abspath(DATA_DIR) # Make sure path is absolute so that symlinks work
    os.mkdir(WORKING_DIR)

    ############################
    ### Create model parameters
    #############################

    target_folder = Categorical('target_folder', ['runs-lama', 'good-operators-unit'], default="good-operators-unit")
    layers_num = Categorical('layers_num', [3, 5, 7], default=3)
    hidden_size = Categorical('hidden_size', [64, 128, 256, 512], default=256)
    conv_type = Categorical('conv_type', ['SAGEConv'], default='SAGEConv')
    aggr = Categorical('aggr', ['mean', 'max'], default='mean')
    optimizer = Categorical('optimizer', ['Adam', 'RMSprop'], default='Adam')
    lr = Categorical('lr', [0.001, 0.01, 0.005], default=0.001)

    parameters = [target_folder, layers_num, hidden_size, conv_type, aggr, optimizer, lr]
 



    cs = ConfigurationSpace(seed=2023) # Fix seed for reproducibility
    cs.add_hyperparameters(parameters)
    # cs.add_conditions(conditions)

    evaluator = Eval (ROOT, DATA_DIR, WORKING_DIR, domain_file, instance_dir)


    print ([ins for ins in instances_with_features])
    print(instances_with_features)
    scenario = Scenario(
        configspace=cs, deterministic=True,
        output_directory=os.path.join(WORKING_DIR, 'smac'),
        walltime_limit=walltime_limit,
        n_trials=n_trials,
        n_workers=n_workers,
        instances=[ins for ins in instances_with_features],
        instance_features=instances_with_features
    )
    # Use SMAC to find the best configuration/hyperparameters
    smac = HyperparameterOptimizationFacade(scenario, evaluator.target_function)
    incumbent = smac.optimize()

    # print("Chosen configuration: ", incumbent)

    model_setting, target_operator = parse_config(incumbent)
    
    print("Chosen model settings: ", model_setting)



    # if 'trained' in  incumbent['queue_type']:
    #     candidate_models.copy_model_to_folder(incumbent, os.path.join(WORKING_DIR, 'incumbent'), symlink=False )
    # else:
    #     os.mkdir(os.path.join(WORKING_DIR, 'incumbent'))

    # with open(os.path.join(WORKING_DIR, 'incumbent', 'config'), 'w') as config_file:
    #     json.dump(incumbent.get_dictionary(), config_file)
        #config_file.write(f"--alias {incumbent['alias']} --grounding-queue {incumbent['queue_type']}")
