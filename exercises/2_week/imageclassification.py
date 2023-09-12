import numpy as np
import os
import random
import torch
from torch import nn
import torch.nn.functional as F
import tqdm
import torch
import torchvision
import torchvision.transforms as transforms
from vit import ViT

import matplotlib.pyplot as plt
import seaborn as sns
import cv2

def plot_attention_maps(image, attention_map, patch_size):
    # Resize the attention map to match the image size
    image = np.transpose(np.transpose(image),axes=[1,0,2])
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    
    attention_map = attention_map.squeeze().cpu().numpy()
    attention_map = np.transpose(np.transpose(attention_map),axes=[1,0,2])
    attention_map = np.mean(attention_map, axis=2)  # Take the mean across all heads
    
    # Upscale the attention map to the original image size
    H, W, _ = image.shape
    attention_map_resized = cv2.resize(attention_map, (W, H), interpolation=cv2.INTER_LINEAR)
    
    # Normalize the attention map
    attention_map_resized -= attention_map_resized.min()
    attention_map_resized /= attention_map_resized.max()

    # Create a heatmap overlay
    heatmap = cv2.applyColorMap(np.uint8(255 * attention_map_resized), cv2.COLORMAP_JET)
    
    # Rotate the heatmap by 90 degrees
    heatmap = cv2.transpose(heatmap)

    # Display the rotated overlaid image
    plt.imshow(heatmap)
    plt.axis('off')
    plt.show()
    
    return heatmap


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def select_two_classes_from_cifar10(dataset, classes):
    idx = (np.array(dataset.targets) == classes[0]) | (np.array(dataset.targets) == classes[1])
    dataset.targets = np.array(dataset.targets)[idx]
    dataset.targets[dataset.targets==classes[0]] = 0
    dataset.targets[dataset.targets==classes[1]] = 1
    dataset.targets= dataset.targets.tolist()  
    dataset.data = dataset.data[idx]
    return dataset

def prepare_dataloaders(batch_size, classes=[3, 7]):
    # TASK: Experiment with data augmentation
    train_transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    test_transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=test_transform)

    # select two classes 
    trainset = select_two_classes_from_cifar10(trainset, classes=classes)
    testset = select_two_classes_from_cifar10(testset, classes=classes)

    # reduce dataset size
    trainset, _ = torch.utils.data.random_split(trainset, [5000, 5000])
    testset, _ = torch.utils.data.random_split(testset, [1000, 1000])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=False
    )
    return trainloader, testloader, trainset, testset


def main(image_size=(32,32), patch_size=(4,4), channels=3, 
         embed_dim=128, num_heads=4, num_layers=4, num_classes=2,
         pos_enc='learnable', pool='cls', dropout=0.3, fc_dim=None, 
         num_epochs=20, batch_size=16, lr=1e-4, warmup_steps=625,
         weight_decay=1e-3, gradient_clipping=1, num_examples=3
         
    ):

    loss_function = nn.CrossEntropyLoss()

    train_iter, test_iter, _, _ = prepare_dataloaders(batch_size=batch_size)

    model = ViT(image_size=image_size, patch_size=patch_size, channels=channels, 
                embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers,
                pos_enc=pos_enc, pool=pool, dropout=dropout, fc_dim=fc_dim, 
                num_classes=num_classes
    )

    if torch.cuda.is_available():
        model = model.to('cuda')

    opt = torch.optim.AdamW(lr=lr, params=model.parameters(), weight_decay=weight_decay)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / warmup_steps, 1.0))

    # training loop
    for e in range(num_epochs):
        print(f'\n epoch {e}')
        model.train()
        for image, label in tqdm.tqdm(train_iter):
            if torch.cuda.is_available():
                image, label = image.to('cuda'), label.to('cuda')
            opt.zero_grad()
            out = model(image)
            loss = loss_function(out, label)
            loss.backward()
            # if the total gradient vector has a length > 1, we clip it back down to 1.
            if gradient_clipping > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            opt.step()
            sch.step()
            
        # Evaluate on validation set and visualize attention maps for example images
        with torch.no_grad():
            model.eval()
            tot, cor = 0.0, 0.0
            example_counter = 0  # Counter for example images
            for image, label in test_iter:
                if torch.cuda.is_available():
                    image, label = image.to('cuda'), label.to('cuda')
                out = model(image).argmax(dim=1)
                tot += float(image.size(0))
                cor += float((label == out).sum().item())
                
                heatmapl = []
                # Visualize attention maps for the first few example images
                if example_counter < num_examples:
                    attention_map = model.get_attention_maps(image)
                    for i in range(1):
                        heatmap = plot_attention_maps(image[i].cpu().numpy(), attention_map[i],patch_size)
                        heatmapl.append(heatmap)
                        example_counter += 1

            acc = cor / tot
            print(f'Validation accuracy: {acc:.3}')            






from itertools import product

def generate_combinations(hyperparameters):
    keys = hyperparameters.keys()
    values = (hyperparameters[key] for key in keys)
    combinations = [dict(zip(keys, combination)) for combination in product(*values)]
    return combinations


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]= str(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print(f"Model will run on {device}")
    set_seed(seed=1)
    main()

