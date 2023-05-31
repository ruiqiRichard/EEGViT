from models.EEGViT_pretrained import EEGViT_pretrained
from models.EEGViT import EEGViT_raw
from models.ViTBase import ViTBase
from models.ViTBase_pretrained import ViTBase_pretrained
from helper_functions import split
from dataset.EEGEyeNet import EEGEyeNetDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np

'''
models: EEGViT_pretrained; EEGViT_raw; ViTBase; ViTBase_pretrained
'''
model = EEGViT_pretrained()
EEGEyeNet = EEGEyeNetDataset('./dataset/Position_task_with_dots_synchronised_min.npz')
batch_size = 64
n_epoch = 15
learning_rate = 1e-4

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)


def train(model, optimizer, scheduler = None):
    '''
        model: model to train
        optimizer: optimizer to update weights
        scheduler: scheduling learning rate, used when finetuning pretrained models
    '''
    torch.cuda.empty_cache()
    train_indices, val_indices, test_indices = split(EEGEyeNet.trainY[:,0],0.7,0.15,0.15)  # indices for the training set
    print('create dataloader...')
    
    train = Subset(EEGEyeNet,indices=train_indices)
    val = Subset(EEGEyeNet,indices=val_indices)
    test = Subset(EEGEyeNet,indices=test_indices)

    train_loader = DataLoader(train, batch_size=batch_size)
    val_loader = DataLoader(val, batch_size=batch_size)
    test_loader = DataLoader(test, batch_size=batch_size)

    if torch.cuda.is_available():
        gpu_id = 0  # Change this to the desired GPU ID if you have multiple GPUs
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)  # Wrap the model with DataParallel
    print("HI")

    model = model.to(device)
    criterion = criterion.to(device)

    # Initialize lists to store losses
    train_losses = []
    val_losses = []
    test_losses = []
    print('training...')
    # Train the model
    for epoch in range(n_epoch):
        model.train()
        epoch_train_loss = 0.0

        for i, (inputs, targets, index) in tqdm(enumerate(train_loader)):
            # Move the inputs and targets to the GPU (if available)
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Compute the outputs and loss for the current batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets.squeeze())

            # Compute the gradients and update the parameters
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

            # Print the loss and accuracy for the current batch
            if i % 100 == 0:
                print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item()}")

        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)

        # Evaluate the model on the validation set
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for inputs, targets, index in val_loader:
                # Move the inputs and targets to the GPU (if available)
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Compute the outputs and loss for the current batch
                outputs = model(inputs)
                # print(outputs)
                loss = criterion(outputs.squeeze(), targets.squeeze())
                val_loss += loss.item()


            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            print(f"Epoch {epoch}, Val Loss: {val_loss}")

        with torch.no_grad():
            val_loss = 0.0
            for inputs, targets, index in test_loader:
                # Move the inputs and targets to the GPU (if available)
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Compute the outputs and loss for the current batch
                outputs = model(inputs)

                loss = criterion(outputs.squeeze(), targets.squeeze())
                val_loss += loss.item()

            val_loss /= len(test_loader)
            test_losses.append(val_loss)

            print(f"Epoch {epoch}, test Loss: {val_loss}")

        if scheduler is not None:
            scheduler.step()

if __name__ == "__main__":
    train(model,optimizer=optimizer, scheduler=scheduler)