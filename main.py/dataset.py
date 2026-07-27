import os
import random
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import (
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    IMAGE_SIZE,
    BATCH_SIZE
)

# -----------------------------------
# Image Transformations
# -----------------------------------

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class TripletDataset(Dataset):
    def __init__(self, image_folder_dataset):
        self.dataset = image_folder_dataset
        self.classes = self.dataset.classes
        
        # Group indices by class
        self.class_to_indices = {i: [] for i in range(len(self.classes))}
        for idx, (_, label) in enumerate(self.dataset.samples):
            self.class_to_indices[label].append(idx)
            
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, index):
        anchor_img, label = self.dataset[index]
        
        # Get positive example from same class
        positive_idx = index
        while positive_idx == index:
            positive_idx = random.choice(self.class_to_indices[label])
        positive_img, _ = self.dataset[positive_idx]
        
        # Get negative example from a different class
        negative_label = random.choice([l for l in self.class_to_indices.keys() if l != label])
        negative_idx = random.choice(self.class_to_indices[negative_label])
        negative_img, _ = self.dataset[negative_idx]
        
        return anchor_img, positive_img, negative_img


# -----------------------------------
# Load Datasets
# -----------------------------------

def get_train_dataset():
    base_dataset = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=train_transform
    )
    return TripletDataset(base_dataset)


def get_val_dataset():
    base_dataset = datasets.ImageFolder(
        root=VAL_DIR,
        transform=test_transform
    )
    return TripletDataset(base_dataset)


def get_test_dataset():
    base_dataset = datasets.ImageFolder(
        root=TEST_DIR,
        transform=test_transform
    )
    return TripletDataset(base_dataset)


# -----------------------------------
# Create DataLoaders
# -----------------------------------

def get_train_loader():

    dataset = get_train_dataset()

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    return loader


def get_val_loader():

    dataset = get_val_dataset()

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return loader


def get_test_loader():

    dataset = get_test_dataset()

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return loader


# -----------------------------------
# Get Class Names
# -----------------------------------

def get_classes():

    dataset = get_train_dataset()

    return dataset.classes


# -----------------------------------
# Number of Classes
# -----------------------------------

def get_num_classes():

    dataset = get_train_dataset()

    return len(dataset.classes)