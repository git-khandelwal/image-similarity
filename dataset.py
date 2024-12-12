from torch.utils.data import Dataset, DataLoader
import os
import torch
import cv2
from torchvision import transforms
import numpy as np


class ImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_files = []
        self.labels = []
        for label, animal_dir in enumerate(os.listdir(folder_path)):
            animal_path = os.path.join(folder_path, animal_dir)
            if os.path.isdir(animal_path):
                self.image_files.extend([os.path.join(animal_path, f) for f in os.listdir(animal_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
                self.labels.extend([label] * len(os.listdir(animal_path)))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, img_path, label  # Returning image, image_path and label for storing into db


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((56, 56)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

animals = [
    "antelope", "badger", "bat", "bear", "bee", "beetle", "bison", "boar", "butterfly", "cat",
    "caterpillar", "chimpanzee", "cockroach", "cow", "coyote", "crab", "crow", "deer", "dog",
    "dolphin", "donkey", "dragonfly", "duck", "eagle", "elephant", "flamingo", "fly", "fox", "goat",
    "goldfish", "goose", "gorilla", "grasshopper", "hamster", "hare", "hedgehog", "hippopotamus",
    "hornbill", "horse", "hummingbird", "hyena", "jellyfish", "kangaroo", "koala", "ladybugs",
    "leopard", "lion", "lizard", "lobster", "mosquito", "moth", "mouse", "octopus", "okapi",
    "orangutan", "otter", "owl", "ox", "oyster", "panda", "parrot", "pelecaniformes", "penguin",
    "pig", "pigeon", "porcupine", "possum", "raccoon", "rat", "reindeer", "rhinoceros", "sandpiper",
    "seahorse", "seal", "shark", "sheep", "snake", "sparrow", "squid", "squirrel", "starfish",
    "swan", "tiger", "turkey", "turtle", "whale", "wolf", "wombat", "woodpecker", "zebra"
]

animal_array = np.array(animals)

# dataset = ImageDataset(folder_path='animals/animals', transform=transform)
# data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

