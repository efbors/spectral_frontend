import os
import yaml

class YamlConfig:
    def __init__(self, config_file_name):
        assert os.path.exists(config_file_name), f"Config file '{config_file_name}' does not exist"
        assert config_file_name.endswith(".yaml")

        with open(config_file_name, 'r') as f:
            self.cfg_dict = yaml.safe_load(f)

    def get_config(self):
        return self.cfg_dict