import os
import torch
from torch import nn
import pandas as pd
from os.path import join
from PIL import Image
import matplotlib.pyplot as plt


torch.manual_seed(42)
torch.manual_seed(42)
torch.manual_seed(42)

dataset_id = "Cars4GAN"


def find_classes(root, dataset_id):

    classes_list = [entry.name for entry in os.scandir(join(root, "Data", dataset_id)) if entry.is_dir()]
    class2idx = {cls: i for i, cls in enumerate(classes_list)}

    return classes_list, class2idx


class ImageDatasetGAN(nn.Module):

    def __init__(self, root, data_file, dataset_id, manual_transform=None):
        super(ImageDatasetGAN, self).__init__()

        self.data_annotation = pd.read_csv(join(root, "Data", dataset_id, data_file))
        self.transform = manual_transform
        self.classes, self.class2idx = find_classes(root=root, dataset_id=dataset_id)

    def __len__(self):
        return len(self.data_annotation)

    def __getitem__(self, idx):

        path2data = join(self.data_annotation["root"][idx], self.data_annotation["parent"][idx],
                         self.data_annotation["dataset_id"][idx], self.data_annotation["class"][idx],
                         self.data_annotation["name"][idx])

        img, label = Image.open(path2data), self.class2idx[self.data_annotation["class"][idx]]

        if self.transform:
            return self.transform(img), label

        if not self.transform:
            return img, label


if __name__ == "__main__":

    root = r"D:\mreza\TestProjects\Python\DCGAN"
    dataset_id = "Cars4GAN"
    data_file = "annotation.csv"
    car_dataset = ImageDatasetGAN(root=root, dataset_id=dataset_id, data_file=data_file)

    # Show samples of the data
    fig, axes = plt.subplots(nrows=3, ncols=3)
    random_indices = torch.randint(0, car_dataset.__len__(), size=(9, ))
    count = 0
    for i in range(3):
        for j in range(3):
            img, label = car_dataset[random_indices[count].item()]
            axes[i][j].imshow(img)
            axes[i][j].set_title(car_dataset.classes[label])

            count += 1

    fig.tight_layout()
    plt.savefig("data_samples.png")
    plt.show()
