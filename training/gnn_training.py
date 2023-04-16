import os
import sys
from lab.calls.call import Call
from dataclasses import dataclass

CONVOLUTIONS = {
    "SAGEConv",
    "GATConv",
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

OPTIMIZER_CLASSES = {
    "Adam",
    "RMSprop",
    "Adagrad"
}

@dataclass
class PreprocessorSettings:
    gnn_retries: int
    gnn_threshold: float

    def __post_init__(self):
        self.gnn_retries = int(self.gnn_retries)
        self.gnn_threshold = float(self.gnn_threshold)

    def to_parameter_string(self):
        return (f"gnn-retries,{self.gnn_retries},"
                f"gnn-threshold,{self.gnn_threshold}")

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
        self.checks(self.optimizer, OPTIMIZER_CLASSES)
    
    def checks(cls, val, allowed_vals):
        if val not in allowed_vals:
            raise ValueError(f"Value {val} not supported.")

    def to_parameter_string(self):
        return (f"layers_num,{self.layers_num},"
                f"hidden_size,{self.hidden_size},"
                f"conv_type,{self.conv_type},"
                f"aggr,{self.aggr},"
                f"optimizer,{self.optimizer},"
                f"lr,{self.lr}")
    
    def to_parameter_string_comma(self):
        return (f"{self.layers_num},"
                f"{self.hidden_size},"
                f"{self.conv_type},"
                f"{self.aggr},"
                f"{self.optimizer},"
                f"{self.lr}")

def run_step_gnn_learning(REPO_LEARNING, problems_dir, output_dir, training_dir, time_limit=300, memory_limit = 4*1024*1024):
    train_dir = os.path.join(problems_dir, "train")
    test_dir = os.path.join(problems_dir, "test")
    problems = os.listdir(problems_dir)
    import shutil
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    os.mkdir(train_dir)
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.mkdir(test_dir)


    # TODO TEMPORARY
    def split_instances(problems_dir, problems, train_dir, test_dir):
        for problem in problems:
            original_problem_dir = os.path.join(problems_dir, problem)
            dst = os.path.join(train_dir, problem)
            shutil.copytree(original_problem_dir, dst)
            

    split_instances(problems_dir, problems, train_dir, test_dir)
    assert len(os.listdir(train_dir)) == len(problems)
    assert os.listdir(test_dir) == []

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
        Call([sys.executable, f'{REPO_LEARNING}/src/train.py', train_dir, test_dir, output_dir, "--model-settings", setting_str], "train-gnn" ,time_limit=time_limit, memory_limit=memory_limit).wait()

    # Make domain knowledge folder
    DK_DIR = os.path.join(training_dir, "DK")
    if os.path.exists(DK_DIR):
        shutil.rmtree(DK_DIR)
    os.mkdir(DK_DIR)

    # TODO: Selection of the best model here

    # Copy the best model to the DK folder
    best_model = os.path.join(output_dir, "models", "4-64-SAGEConv-mean-Adam-0.001", "0.pt")
    shutil.copy(best_model, os.path.join(DK_DIR, "model.pt"))

    # TODO: Save model settings as string in the DK folder
    with open(os.path.join(DK_DIR, "model_settings.txt"), "w") as f:
        f.write(adam.to_parameter_string_comma())

    # TODO: Save the preprocessor settings as string to the DK folder
    preprocessor_settings = PreprocessorSettings(gnn_retries=3, gnn_threshold=0.5)
    with open(os.path.join(DK_DIR, "preprocessor_settings.txt"), "w") as f:
        f.write(preprocessor_settings.to_parameter_string())
        
    # DK_DIR into zip file
    shutil.make_archive(DK_DIR, 'zip', DK_DIR)
