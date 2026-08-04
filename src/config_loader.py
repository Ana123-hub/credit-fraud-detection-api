import os
import yaml

def load_config():
    """
    Locates and loads config.yaml from the project root directory,
    returning configuration dictionary with absolute path resolution.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, ".."))
    config_path = os.path.join(root_dir, "config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: '{config_path}'")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Convert relative paths in config.yaml to absolute paths
    for key, val in config["paths"].items():
        config["paths"][key] = os.path.abspath(os.path.join(root_dir, val))

    return config

# Export global singleton
CONFIG = load_config()