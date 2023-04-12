import os
import sys
from lab.calls.call import Call
import shutil
from dataclasses import dataclass, field

file_path = str
dir_path = str

CONVOLUTIONS = {
    "SAGEConv",
    "GATConv"
    "GCNConv"
}

AGGREGATIONS = {
    "mean",
    "max",
    "min",
    "sum",
    "var",
    "median",
    "std"
}

optimizer_classes = {
    "Adam"
    "RMSprop"
    "Adagrad"
}


@dataclass
class ModelSetting:
    layers_num: int
    hidden_size: int
    conv_type: str
    aggr: str
    optimizer: str
    lr: float

    def __post_init__(self):
        self.layers_num = int(self.layers_num)
        self.hidden_size = int(self.hidden_size)
        self.lr = float(self.lr)

        self.checks(self.conv_type, CONVOLUTIONS)
        self.checks(self.aggr, AGGREGATIONS)
        self.checks(self.optimizer, optimizer_classes)
    
    def checks(cls, val, allowed_vals):
        if val not in allowed_vals:
            raise ValueError(f"Value {val} not supported.")

    def to_parameter_string(self):
        return (f"'layers_num,{self.layers_num},"
                f"hidden_size,{self.hidden_size},"
                f"conv_type,{self.conv_type},"
                f"aggr,{self.aggr},"
                f"optimizer,{self.optimizer},"
                f"lr,{self.lr}'")

def run_step_gnn_learning(REPO_LEARNING, WORKING_DIR, domain_file, time_limit=300, memory_limit = 4*1024*1024):
    raise NotImplementedError
    def get_train_instances(train_dir):
        return [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith(".pddl")]

    def get_test_instances(test_dir=None):
        if test_dir is None:
            return []
        else:
            pass
    
    def get_executable(repo_learning, file_name, **options):
        pass

    def generate_model_settings():
        # This will be predefined:
        
        adam = ModelSetting(
            layers_num=4,
            hidden_size=64,
            conv_type="SAGEConv",
            aggr="mean",
            optimizer="Adam",
            lr=0.001
        )

        sgd = ModelSetting(
            layers_num=4,
            hidden_size=64,
            conv_type="SAGEConv",
            aggr="mean",
            optimizer="RMSprop",
            lr=0.001
        )

        settings = [adam.to_parameter_string(), sgd.to_parameter_string()]
        for setting_str in settings:
            Call("python src/train.py train test workspace --model-settings"+ " "+setting_str)


    # models_dir




# for each problem passed as the --path/problem.pddl
# we put them to the directory: data/instances_training/problem.pddl
# This will return good_operators file for each data/instances_trainng/name.pddl --> data/good-operators-unit/name



# ja proponuje pod training zrobic kolejny plik tkroy sie nazywa make gnn files, ktory uruchomimy z learn.py