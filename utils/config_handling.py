import yaml, os, glob

def load_config(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config

def create_config():
    pass