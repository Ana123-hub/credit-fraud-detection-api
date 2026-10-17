import os
from pathlib import Path
import yaml

def load_config():
    """
    Load project configuration and resolve directory/data paths.
    """
    default_root = Path(__file__).resolve().parent.parent

    project_root = Path(
        os.environ.get("CREDIT_FRAUD_PROJECT_ROOT", str(default_root))
    ).resolve()
    config_path = project_root / "config.yaml"

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Config file not found at: '{config_path}'"
        )


    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict) or not isinstance(
        config.get("paths"), dict
    ):
         raise ValueError(
            "config.yaml must contain a valid 'paths' mapping."
        )

    # Resolve only actual paths; Artifact filenames remain filenames.
    for key in ("raw_data", "processed_dir", "models_dir"):
        if key not in config["paths"]:
            raise KeyError(
                f"Missing required configuration path: '{key}'"
            )
        configured_path = Path(config["paths"][key])

        if not configured_path.is_absolute():
            configured_path = project_root / configured_path

        config["paths"][key] = str(configured_path.resolve())

    return config

# Export
CONFIG = load_config()