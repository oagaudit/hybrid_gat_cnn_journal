import os
import yaml

def get_project_root():
    current = os.path.abspath(os.path.dirname(__file__))
    while current != "/":
        if "config" in os.listdir(current):
            return current
        current = os.path.dirname(current)
    raise Exception("Project root not found")

def load_config():
    project_root = get_project_root()
    config_path = os.path.join(project_root, "config", "config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

CONFIG = load_config()