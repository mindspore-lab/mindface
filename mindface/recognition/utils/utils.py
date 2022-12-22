"""
Tools for training
"""
import os
import json
import yaml

__all__ = ["read_yaml", "ObsToEnv", "C2netMultiObsToEnv", "EnvToObs"]

def read_yaml(path):
    """
    read_yaml
    """
    with open(path, 'r', encoding='utf-8') as file:
        string = file.read()
        info_dict = yaml.safe_load(string)

    return info_dict

def ObsToEnv(obs_data_url, data_dir):
    """
    ObsToEnv.
    """
    import moxing as mox
    try:
        mox.file.copy_parallel(obs_data_url, data_dir)
        print("Successfully Download {} to {}".format(obs_data_url, data_dir))
    except Exception as e:
        print('moxing download {} to {} failed: '.format(obs_data_url, data_dir) + str(e))

    f = open("/cache/download_input.txt", 'w')
    f.close()
    try:
        if os.path.exists("/cache/download_input.txt"):
            print("download_input succeed")
    except Exception as e:
        print("download_input failed")

def C2netMultiObsToEnv(multi_data_url, data_dir):
    """
    C2netMultiObsToEnv.
    """
    import moxing as mox

    multi_data_json = json.loads(multi_data_url)
    for i in range(len(multi_data_json)):
        zipfile_path = os.path.join(data_dir, multi_data_json[i]["dataset_name"])
        try:
            mox.file.copy(multi_data_json[i]["dataset_url"], zipfile_path)
            print("Successfully Download {} to {}".format(
                  multi_data_json[i]["dataset_url"], zipfile_path))
            #unzip the dataset
            print("unzip -q {} -d {}".format(zipfile_path, data_dir))
            os.system("unzip -q {} -d {}".format(zipfile_path, data_dir))

        except Exception as e:
            print('moxing download {} to {} failed: '.format(
                multi_data_json[i]["dataset_url"], zipfile_path) + str(e))

    f = open("/cache/download_input.txt", 'w')
    f.close()
    try:
        if os.path.exists("/cache/download_input.txt"):
            print("download_input succeed")
    except Exception as e:
        print("download_input failed")

def EnvToObs(train_dir, obs_train_url):
    """
    Copy the output to obs.
    """
    import moxing as mox
    try:
        mox.file.copy_parallel(train_dir, obs_train_url)
        print("Successfully Upload {} to {}".format(train_dir,obs_train_url))
    except Exception as e:
        print('moxing upload {} to {} failed: '.format(train_dir,obs_train_url) + str(e))
