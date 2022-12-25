"""
Tools for training
"""
import os
import json
import yaml
import moxing as mox

__all__ = ["ReadYaml", "ObsToEnv", "C2netMultiObsToEnv", "EnvToObs"]

def ReadYaml(path):
    """
    ReadYaml
    """
    with open(path, 'r', encoding='utf-8') as file:
        string = file.read()
        info_dict = yaml.safe_load(string)

    return info_dict

# pylint: disable=C0103
def ObsToEnv(obs_data_url, data_dir):
    """
    Copy single dataset from obs to inference image.
    """
    try:
        mox.file.copy_parallel(obs_data_url, data_dir)
        print(f"Successfully Download {obs_data_url} to {data_dir}")
    # pylint: disable=W0703
    except Exception as e:
        print(f"moxing download {obs_data_url} to {data_dir} failed: {e}")

    # f = open("/cache/download_input.txt", 'w', encoding="utf-8")
    # f.close()
    try:
        if os.path.exists("/cache/download_input.txt"):
            print("download_input succeed")
    # pylint: disable=W0703
    except Exception as e:
        print(f"download_input failed: {e}")

def C2netMultiObsToEnv(multi_data_url, data_dir):
    """
    C2netMultiObsToEnv.
    """
    multi_data_json = json.loads(multi_data_url)
    for i in enumerate(multi_data_json):
        zipfile_path = os.path.join(data_dir, multi_data_json[i]["dataset_name"])
        try:
            mox.file.copy(multi_data_json[i]["dataset_url"], zipfile_path)
            print(f"Successfully Download to {zipfile_path}")
            #unzip the dataset
            print(f"unzip -q {zipfile_path} -d {data_dir}")
            os.system(f"unzip -q {zipfile_path} -d {data_dir}")
        # pylint: disable=W0703
        except Exception as e:
            print(f"moxing download to {zipfile_path} failed: {e}")

    # f = open("/cache/download_input.txt", 'w', encoding="utf-8")
    # f.close()
    try:
        if os.path.exists("/cache/download_input.txt"):
            print("download_input succeed")
    # pylint: disable=W0703
    except Exception as e:
        print(f"download_input failed: {e}")

def EnvToObs(train_dir, obs_train_url):
    """
    Copy the output to obs.
    """
    try:
        mox.file.copy_parallel(train_dir, obs_train_url)
        print(f"Successfully Upload {train_dir} to {obs_train_url}")
    # pylint: disable=W0703
    except Exception as e:
        print(f"moxing upload {train_dir} to {obs_train_url} failed: {e}")
