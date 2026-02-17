import wandb
import yaml

PROJECT_NAME = "scl-sweep"

if __name__ == "__main__":
    yaml_file_path = "scripts/sweep.yaml"

    with open(yaml_file_path, "r") as file:
        sweep_configuration = yaml.safe_load(file)

    sweep_id = wandb.sweep(
        sweep=sweep_configuration,
        project=PROJECT_NAME
    )

    wandb.agent(sweep_id, count=30)
