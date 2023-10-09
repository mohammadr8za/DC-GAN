import os
from os.path import join, split
import pandas as pd
import glob
from pathlib import Path


def make_annotation(root, dataset_id):

    directory_path = join(root, "Data", dataset_id)
    data_list = glob.glob(directory_path + r"/**/*.jpg", recursive=True)

    data_dict = {"root": [root for i in range(len(data_list))],
                 "parent": ["Data" for i in range(len(data_list))],
                 "dataset_id": [dataset_id for i in range(len(data_list))],
                 "class": [],
                 "name": []}

    for data in data_list:
        data_dict["name"].append(split(data)[1])
        data_dict["class"].append(split(split(data)[0])[1])

    df = pd.DataFrame(data_dict)
    df.to_csv(join(root, "Data", dataset_id, "annotation.csv"), index=False)


if __name__ == "__main__":

    root = r"D:\mreza\TestProjects\Python\DCGAN"
    dataset_id = "Cars4GAN"
    make_annotation(root=root, dataset_id=dataset_id)


