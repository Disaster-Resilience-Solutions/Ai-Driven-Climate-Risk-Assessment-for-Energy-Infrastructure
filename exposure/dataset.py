import os, cv2, torch
import numpy as np
import glob as glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from xml.etree import ElementTree as et
from torch.utils.data import Dataset, DataLoader

from configuration import train_dir, val_dir, classes, width, height, batch_size
from utils import train_transform, valid_transform, collate_fn


# Customised class for specific image data

class OnshoreWindTurbines(Dataset):

    def __init__(self, dir_path, width, height, classes, transforms=None):
        self.transforms = transforms
        self.dir_path = dir_path
        self.height = height
        self.width = width
        self.classes = classes

        # self.image_paths = glob.glob(f"{self.dir_path}/*.png")
        self.image_paths = glob.glob(self.dir_path + '/*.png')
        self.all_images = [image_path.split(os.path.sep)[-1] for image_path in self.image_paths]
        self.all_images = sorted(self.all_images)

    def __getitem__(self, idx):

        # Iterate through png files in custom directory

        image_name = self.all_images[idx]
        image_path = os.path.join(self.dir_path, image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0

        # Iterate through xml files in custom directory

        annot_filename = image_name[:-4] + '.xml'
        annot_file_path = os.path.join(self.dir_path, annot_filename)
        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()
        image_width = image.shape[1]
        image_height = image.shape[0]

        # Designate bounding coordinate labels to a class - dict format

        for member in root.findall('object'):
            labels.append(self.classes.index(member.find('name').text))
            xmin = int(member.find('bndbox').find('xmin').text)
            xmax = int(member.find('bndbox').find('xmax').text)
            ymin = int(member.find('bndbox').find('ymin').text)
            ymax = int(member.find('bndbox').find('ymax').text)

            xmin_final = (xmin / image_width) * self.width
            xmax_final = (xmax / image_width) * self.width
            ymin_final = (ymin / image_height) * self.height
            yamx_final = (ymax / image_height) * self.height

            boxes.append([xmin_final, ymin_final, xmax_final, yamx_final])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Specify coordinate class labels as target features

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        if self.transforms:
            sample = self.transforms(image=image_resized,
                                     bboxes=target['boxes'],
                                     labels=labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])

        return image_resized, target

    def __len__(self):
        return len(self.all_images)


# Specify transformed datasets

def create_train_dataset():
    train_dataset = OnshoreWindTurbines(train_dir, width, height, classes, train_transform())
    return train_dataset


def create_validation_dataset():
    valid_dataset = OnshoreWindTurbines(val_dir, width, height, classes, valid_transform())
    return valid_dataset


# PyTorch compatible dataloader functions

def create_train_loader(train_dataset, num_workers):
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              collate_fn=collate_fn)
    return train_loader


def create_validation_loader(valid_dataset, num_workers):
    valid_loader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers,
                              collate_fn=collate_fn)
    return valid_loader


# Call customised baseline dataset

if __name__ == '__main__':
    dataset = OnshoreWindTurbines(train_dir, width, height, classes)


# Bounding box plotting function

def plot_img_bbox(image, target):
    fig, a = plt.subplots(1, 1)
    fig.set_size_inches(5, 5)
    if isinstance(image, torch.Tensor):
        image = torch.transforms.ToPILImage()(image).convert('RGB')
    a.imshow(image)
    boxes = target['boxes']
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.tolist()
    for box in (boxes):
        x, y, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]
        rect = patches.Rectangle((x, y),
                                 width, height,
                                 linewidth=3,
                                 edgecolor='r',
                                 facecolor='none')
        a.add_patch(rect)
    plt.show()
    print("Total detected wind turbines: ", len(boxes))

img_samples = 3
for i in range(img_samples):
    image, target = dataset[i]
    plot_img_bbox(image, target)
