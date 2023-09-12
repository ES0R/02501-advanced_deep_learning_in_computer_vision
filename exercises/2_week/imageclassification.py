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

import numpy as np
import matplotlib.pyplot as plt
import cv2

import numpy as np
import matplotlib.pyplot as plt
import cv2

def plot_attention_maps(model, image, attention_map):
    # Ensure the model is in evaluation mode
    model.eval()

    # Convert the attention map to numpy and take its mean if it has multiple channels (e.g., multiple heads)
    if len(attention_map.shape) > 2:
        attention_map = attention_map.mean(dim=0)
    attention_map = attention_map.cpu().numpy()

    # Ensure the image is in the shape (H, W, C)
    if image.shape[0] == 3:  # If image is in (C, H, W) format
        image = np.transpose(image, (1, 2, 0))

    # 1. Rescale the Attention Map
    attention_map_resized = attention_map.copy()
    attention_map_resized -= attention_map_resized.min()
    attention_map_resized /= attention_map_resized.max()

    # Resize the attention map to match the image size
    attention_map_resized = cv2.resize(attention_map_resized, (image.shape[1], image.shape[0]))

    # 2. Adjust the Alpha Value
    alpha = 0.999  # You can experiment with this value
    overlayed_image = ((1 - alpha) * image + alpha * attention_map_resized[..., None] * 255).astype(np.uint8)

    # 3. Visualize the Rescaled Attention Map
    #plt.imshow(attention_map_resized, cmap='viridis')
    #plt.colorbar()
    #plt.title("Rescaled Attention Map")
    #plt.axis('off')
    #plt.show()

    # 4. Visualize the Original Attention Map
    plt.imshow(attention_map, cmap='viridis')
    plt.colorbar()
    plt.title("Original Attention Map")
    plt.axis('off')
    plt.show()

    # 5. Ensure Proper Image Values
    #image = (image * 255).astype(np.uint8)

    # Display the original image
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    plt.show()

    # Display the overlaid image
    plt.imshow(overlayed_image)
    plt.axis('off')
    plt.show()

    return overlayed_image








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
        running_loss = 0.0
        for i, (image, label) in enumerate(tqdm.tqdm(train_iter)):
            if torch.cuda.is_available():
                image, label = image.to('cuda'), label.to('cuda')
            opt.zero_grad()
            out = model(image)
            loss = loss_function(out, label)
            loss.backward()
            # Gradient clipping
            if gradient_clipping > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            opt.step()
            sch.step()
            running_loss += loss.item()
            if i % 100 == 99:  # Print average loss every 100 batches
                print(f"[{e + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0
        
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
                
                # Visualize attention maps for the first few example images
                if example_counter < num_examples:
                    attention_map = model.get_attention_maps(image)
                    for i in range(image.size(0)):
                        heatmap = plot_attention_maps(model, image[i].cpu().numpy(), attention_map[i])
                        example_counter += 1
                if example_counter >= num_examples:
                    break  # Exit the loop once we've visualized the desired number of examples

            acc = cor / tot
            print(f'Validation accuracy: {acc:.3}')            
         



if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"]= str(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print(f"Model will run on {device}")
    set_seed(seed=1)
    main()


