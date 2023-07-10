import yaml


def parse_config(path: str, tissue_type: str, scenario: str) -> dict:
    """
    Parse yaml file

    Params:
        path: str, path to config

    Returns:
        cfg: dict, dictionary with new params
    """

    with open(path) as f:
        data = yaml.load(f, Loader=yaml.Loader)
        cfg = data[tissue_type][scenario]

    return cfg
