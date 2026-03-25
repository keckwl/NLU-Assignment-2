from generate_names_dataset import generate_training_file
from train import run_training
from evaluate import evaluate_all

generate_training_file("TrainingNames.txt")
run_training(names_path="TrainingNames.txt", output_dir="outputs")
evaluate_all(names_path="TrainingNames.txt", output_dir="outputs")
