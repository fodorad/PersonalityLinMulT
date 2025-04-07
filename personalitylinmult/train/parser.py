import argparse
import yaml
from pprint import pprint


def argparser(verbose=True) -> dict:
    args, unknown = setup_argparse()
    config = load_and_merge_configs(args, unknown)
    if verbose: pprint(config)
    return config


def parse_additional_args(unknown_args):
    """Parses additional key-value pairs from unknown arguments."""
    additional_args = {}
    if len(unknown_args) % 2 == 1:
        raise ValueError("Only key-value pairs are supported for additional arguments.")
    for i in range(0, len(unknown_args), 2):
        key = unknown_args[i].lstrip("--") # Remove leading '--'
        value = unknown_args[i + 1]
        # Try to cast to int or float if possible
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass
        additional_args[key] = value
    return additional_args


def load_yaml_config(file_path):
    """Loads a YAML configuration file."""
    try:
        with open(file_path, "r") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file '{file_path}' not found.")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file '{file_path}': {e}")


def setup_argparse():
    """Sets up the argument parser and returns parsed arguments."""
    parser = argparse.ArgumentParser(description="Script with improved argparse handling")
    parser.add_argument("--db_config_path", type=str, help="Path to the DB config file")
    parser.add_argument("--model_config_path", type=str, help="Path to the Model config file")
    parser.add_argument("--train_config_path", type=str, help="Path to the Train config file")
    args, unknown = parser.parse_known_args()
    return args, unknown


def load_and_merge_configs(args, unknown) -> dict:
    """Loads YAML configs and merges them with additional CLI arguments."""
    config = {}
    
    # Load YAML configurations if paths are provided
    if args.db_config_path:
        config.update(load_yaml_config(args.db_config_path))
    if args.model_config_path:
        config.update(load_yaml_config(args.model_config_path))
    if args.train_config_path:
        config.update(load_yaml_config(args.train_config_path))
    
    # Parse and merge additional CLI arguments
    additional_args = parse_additional_args(unknown)
    config.update(additional_args)
    
    return config