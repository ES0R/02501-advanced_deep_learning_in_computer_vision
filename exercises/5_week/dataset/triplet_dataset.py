import os
from PIL import Image
from torch.utils.data.dataset import Dataset


class TripletDataset(Dataset):
    """
    Each triplet has its folder with the corresponding 3 images.
    Structure of the triplet dataset:
        triplet_<number>
            -   1.png
            -   2.png
            -   3.png
    """
    def __init__(self, triplets_path, transform):
        self.triplets_path = triplets_path
        self.transform = transform
        self.triplets = list(os.listdir(triplets_path))


    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, index):
        # Make sure you understand the __init__ function and the structure of the data first.
        current_path = os.path.join(self.triplets_path,self.triplets[index])

        
        # TASK 1: Read the triplet and make sure you use the self.transform
        frame_1 = Image.open(os.path.join(current_path, '1.png'))
        frame_2 = Image.open(os.path.join(current_path, '2.png'))
        frame_3 = Image.open(os.path.join(current_path, '3.png'))

        if self.transform:
            frame_1 = self.transform(frame_1)
            frame_2 = self.transform(frame_2)
            frame_3 = self.transform(frame_3)

        return frame_1, frame_2, frame_3
