from torch.utils.data import Dataset, Subset
import imageio
import os
import torchvision
import torch
from torchvision.transforms import v2

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim

import argparse
from src.model import Eye_Fixation
from src.dataset import FixationDataset
import wandb

def evaluate(model, data_loader_test, device):
    """Here in the evaluation function, we will evaluate the model on the test dataset. We cant calulate 
    the accuracy here as we dont have the ground truth in the test set. Hence we do:

    Recall that the FCN outputs log probabilities of predicted fixation probabilities. 
    To visualize these, you should apply the torch.sigmoid activation function to the outputs. 
    This will appropriately scale them to the range [0, 1], after which the predictions can be visualized as above.

    Args:
        model (_type_): _description_
        data_loader_test (_type_): _description_
        device (_type_): _description_

    Returns:
        _type_: _description_
    """
    correct = 0
    total = 0
    loss_val = 0
    # evaluate after each epoch
    with torch.no_grad():
        for images, labels in data_loader_test:
            xb = images.to(device)
            yb = labels.to(device)

            # model prediction
            pred = model(xb)
            loss_val += loss_fcn(pred, yb)

            predictions = torch.argmax(pred, dim=-1)

            # Update counters
            total += yb.size(0)
            correct += (predictions == yb).sum().item()

    # Calculate accuracy
    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}, Loss: {loss_val/total:.4f}")

    return accuracy

def train(epochs, model, loss_fcn, optimizer, data_loader_train, data_loader_val, device):
     # training loop
    epochs = epochs
    for epoch in range(epochs):
        model.train()
        print(f"Epoch: {epoch}")
        # for step, (img) in enumerate(data_loader_train): # over the sampling dimension
        for step in range(0, len(data_loader_train)):
            xb = one_batch["image"].to(device) 
            yb = one_batch["fixation"].to(device)

            # model prediction
            pred = model(xb)
            # print(f"Shape of pred: {pred.shape}, Shape of yb: {yb.shape}, Input Shape: {xb.shape}")
            # print(f"Max value of pred: {torch.max(pred)}, Min value of pred: {torch.min(pred)}, Max value of yb: {torch.max(yb)}, Min value of yb: {torch.min(yb)}, Max value of xb: {torch.max(xb)}, Min value of xb: {torch.min(xb)}")


            # loss
            loss = loss_fcn(pred, yb)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if (step+1) % 1 == 0:
                print(f"Batch Number: {step } -- {len(data_loader_train)}, Loss: {loss.item()}")
        
        correct = 0
        total = 0
        loss_val = 0
        # evaluate after each epoch
        # with torch.no_grad():
        #     for img in data_loader_val:
        #         xb = img["images"].to(device)
        #         yb = img["fixation"].to(device)

        #         # model prediction
        #         pred = model(xb)
        #         loss_val += loss_fcn(pred, yb)

        #         predictions = torch.argmax(pred, dim=-1)

        #         # Update counters
        #         total += yb.size(0)
        #         correct += (predictions == yb).sum().item()

        # Calculate accuracy
        # accuracy = correct / total
        # print(f"Accuracy: {accuracy:.4f}, Loss: {loss_val/total:.4f}")

        print(f"Saving model after epoch {epoch}.")
        # save model after each epoch
        torch.save(
            {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr': optimizer.param_groups[0]['lr'],
            }, f"sailancy_model/sailancy_model_epoch_{epoch}.pt"
        )


if __name__ == "__main__":
    # create directory to save model
    os.makedirs("sailancy_model", exist_ok=True)

    # parse args
    parser = argparse.ArgumentParser(
                    prog='sailancy paramter training.',
                    description='Train sailancy model.',
                    epilog='Choose device.')
    
    parser.add_argument('--device', type=bool, default=False,
                    help='Choose device')
    parser.add_argument('--resume_training', type=int, default=None,
                    help='Choose device')
    parser.add_argument('--epochs', type=int, default=10,
                    help='Choose epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')

    args = parser.parse_args()

    wandb.init(
        project="Saliency-map",
        
        # track hyperparameters and run metadata
        config={
        "device": args.device,
        "architecture": args.resume_training,
        "epochs": args.epochs,
        }
    )

    
    # choose backend
    device = "mps" if (torch.backends.mps.is_available() and args.device) else "cpu"
    print("Model backend:", device)


    # load model, optimizer and state dict
    if args.resume_training is not None: 
        print(f"Resuming training from epoch: {args.resume_training}; Loading Model: sailancy_model_epoch_{args.resume_training}")
        checkpoint = torch.load(f"sailancy_model/sailancy_model_epoch_{args.resume_training}.pt")
        model = Eye_Fixation()
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = optim.SGD(model.parameters(), lr=checkpoint['lr'], momentum=0.9)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model = model.to(device)
        epochs = args.epochs - checkpoint['epoch']
    else: # create instance of model
        model = Eye_Fixation()
        model = model.to(device)
        epochs = args.epochs

    # create loss and optimizer
    loss_fcn = F.binary_cross_entropy_with_logits
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    # Load the dataset
    batch_size_train = args.batch_size
    batch_size_test = args.batch_size * 2

    # tranformation
    transform = v2.Compose([
        v2.ToTensor() # convert the image to a tensor with values between 0 and 1
    ])

    sailancy_train_dataset = FixationDataset(root_dir="cv2_project_data", image_file="cv2_project_data/train_images.txt", 
						   fixation_file="cv2_project_data/train_fixations.txt", image_transform=transform, fixation_transform=transform)
    
    sailancy_val_dataset = FixationDataset(root_dir="cv2_project_data", image_file="cv2_project_data/val_images.txt", 
						   fixation_file="cv2_project_data/val_fixations.txt", image_transform=transform, fixation_transform=transform)


    data_loader_train = torch.utils.data.DataLoader(sailancy_train_dataset,
											batch_size=batch_size_train,
											shuffle=True)

    one_batch = next(iter(data_loader_train)) # get one batch for testing if the model is working
    
    data_loader_val = torch.utils.data.DataLoader(sailancy_val_dataset,
											batch_size=batch_size_test,
											shuffle=False)
    

    import time
    start_time = time.time()
    train(epochs, model, loss_fcn, optimizer, one_batch, data_loader_val, device)
    end_time = time.time()

    print(f"Runtime: {end_time - start_time} seconds")