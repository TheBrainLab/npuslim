import argparse
import yaml
from easydict import EasyDict

from rich.console import Console
from rich.table import Table
from rich.text import Text


class Configuration:
    def prepare():
        args = Configuration.get_args()
        conf = Configuration.load_config(args.config)
        conf = Configuration.merge_config(args, conf)
        Configuration.check_valid(conf)
        Configuration.print_config(conf, f"Configuration of {args.config}")
        return conf

    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser(description="NpuSlim")
        parser.add_argument("-c", "--config", type=str, required=True)
        parser.add_argument("--model-path", type=str, default=None)
        parser.add_argument("--save-path", type=str, default=None)
        # parser.add_argument("--multi-nodes", action="store_true")
        args = parser.parse_args()
        return args

    @staticmethod
    def load_config(path):
        with open(path, 'r') as f:
            config = EasyDict(yaml.safe_load(f))
        return config

    @staticmethod
    def merge_config(args, conf):    
        if args.model_path is not None:
            conf.model.model_path = args.model_path
        if args.save_path is not None:
            conf.metadata.save_path = args.save_path
        return conf

    @staticmethod
    def check_valid(config):
        pass
    
    @staticmethod
    def print_config(config: dict, title: str = "Configuration"):
        console = Console()
        table = Table(show_header=False, show_lines=False, title=f"[bold blue]{title}[/bold blue]")
        table.add_column("Key", style="cyan", no_wrap=False)
        table.add_column("Value", style="magenta", overflow="fold")

        def add_items(d, indent=0):
            for k, v in d.items():
                prefix = "  " * indent + ("- " if indent > 0 else "")
                key_str = f"{prefix}{k}"
                if isinstance(v, dict):
                    table.add_row(Text(key_str, style="bold green"), "")
                    add_items(v, indent=indent+1)
                elif isinstance(v, list):
                    value_str = "\n".join(str(i) for i in v)
                    table.add_row(key_str, value_str)
                else:
                    table.add_row(key_str, str(v))

        for section, subconfig in config.items():
            table.add_row(Text(section.upper(), style="bold green"), "")
            add_items(subconfig, indent=1)
            table.add_section()

        console.print(table)

