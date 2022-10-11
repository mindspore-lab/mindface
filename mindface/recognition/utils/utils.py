"""
Tools for training
"""
import os
import yaml

__all__ = ["read_yaml"]

def read_yaml(path):
    """
    Read yaml.
    """
    with open(path, 'r', encoding='utf-8') as file:
        string = file.read()
        info_dict = yaml.safe_load(string)

    return info_dict

def get_rank_id():
    """
    Get rank id.
    """
    global_rank_id = os.getenv('RANK_ID', '0')
    return int(global_rank_id)
