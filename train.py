from torch.utils.data import Dataset, Subset
import imageio
import os
import torchvision
import torch
from torchvision.transforms import v2, ConvertImageDtype
import numpy as np
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

from torchvision.utils import make_grid
from torchvision.io import read_image


def seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate(model, data_loader_test, device, args):
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
    model.eval()
    predictions_list = []
    # evaluate after each epoch
    with torch.no_grad():
        for img in data_loader_test:
            xb = img["image"].to(device)

            # model prediction
            pred = model(xb)
            predictions = torch.max(pred, dim=0, keepdim=True)[0] # across the batch dimension

            predictions = predictions - predictions.min()
            max_pred = predictions / predictions.max() * 255.0

            max_pred = max_pred.byte()
            predictions_list.append(max_pred) # to sigmoid
    
    # Get the list of image filenames from the dataset
    image_file_names = data_loader_test.dataset.image_files
    predictions_list = torch.cat(predictions_list, dim=0)
    
    # Iterate through the predictions and save them with the appropriate filename
    for step, pred in enumerate(predictions_list):
        # Extract the base name from the original image file name (without extension)
        base_name = os.path.splitext(os.path.basename(image_file_names[step]))[0]
        
        # Construct the new filename using the base name
        new_filename = f"predictions/prediction-{base_name}.png"
        
        # Convert the prediction to a numpy array and save it as an image
        imageio.imwrite(new_filename, pred.squeeze(0).cpu().numpy())

        # Log the predictions to wandb
        pred = pred.float() # convert to float for visualization
        grid_pred = make_grid(pred.unsqueeze(0), nrow=1) # make grid out of one image
        wandb.log({"eval/images": [wandb.Image(grid_pred, caption=new_filename)]}, step=step) # log the image to wandb

    

def train(epochs, model, loss_fcn, optimizer, data_loader_train, data_loader_val, device, args):
     # training loop
    
    step = 0
    for epoch in range(epochs):
        model.train()
        print(f"Epoch: {epoch+1}")
        # for step, (img) in enumerate(data_loader_train): # over the sampling dimension
        for _ in range(100):
            step += 1
            xb = one_batch["image"].to(device) 
            yb = one_batch["fixation"].to(device)

            # model prediction
            pred = model(xb)

            # loss
            loss = loss_fcn(pred, yb)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if (step+1) % args.log_steps == 0:
                print("Step:", step, "Loss:", loss.item())
                # log in wandb
                grid_fix = make_grid(one_batch["fixation"][:8], nrow=4)
                grid_raw = make_grid(one_batch["raw_image"][:8], nrow=4)
                grid_prediction = make_grid(torch.sigmoid(pred)[:8], nrow=4)
                wandb.log({"training/loss": loss.item(),
                            "training/images/tgt_fixation": [wandb.Image(grid_fix)],
                            "training/images/raw_image": [wandb.Image(grid_raw)],
                            "training/images/prediction": [wandb.Image(grid_prediction)]}, step=step)
        
        correct = 0
        total = 0
        loss_val = 0
        # evaluate after each epoch
        with torch.no_grad():
            for img in data_loader_val:
                xb = img["image"].to(device)
                yb = img["fixation"].to(device)

                # model prediction
                pred = model(xb)
                loss_val += loss_fcn(pred, yb)

                predictions = torch.max(pred, dim=0)[0]

                # Update counters
                total += yb.size(0)
                correct += (predictions == yb).sum().item()

        # Calculate accuracy
        accuracy = correct / total
        print(f"Accuracy: {accuracy:.4f}, Loss: {loss_val/total:.4f}")
        wandb.log({"validation/accuracy": accuracy, "validation/loss": loss_val/total})

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
    os.makedirs("predictions", exist_ok=True)

    # parse args
    parser = argparse.ArgumentParser(
                    prog='sailancy paramter training.',
                    description='Train sailancy model.',
                    epilog='Choose device.')
    
    parser.add_argument('--device', type=bool, default=False,
                    help='Choose device')
    parser.add_argument('--resume_training', type=int, default=None,
                    help='Resume training or not.')
    parser.add_argument('--epochs', type=int, default=10,
                    help='Choose epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--log', type=bool, default=False, help='Log to wandb')
    parser.add_argument('--log_steps', type=int, default=10, help='Log steps to wandb')
    parser.add_argument('--seed', type=int, default=123, help='Seed for reproducibility')
    parser.add_argument('--momentum', type=float, default=0.9, help='Opt momentum')
    args = parser.parse_args()
    
    # set seed for reproducibility
    seed(args.seed)


    wandb.init(
        project="Saliency-map",
        
        # track hyperparameters and run metadata
        config={
        "device": args.device,
        "resume_training": args.resume_training,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "log": args.log,
        "log_steps": args.log_steps,
        "momentum": args.momentum,
        },

        mode = "online" if args.log else "disabled"
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
        optimizer = optim.SGD(model.parameters(), lr=checkpoint['lr'], momentum=args.momentum)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model = model.to(device)
        epochs = args.epochs - checkpoint['epoch']
    else: # create instance of model
        model = Eye_Fixation()
        model = model.to(device)
        epochs = args.epochs

    # create loss and optimizer
    loss_fcn = F.binary_cross_entropy_with_logits
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Load the dataset
    batch_size_train = args.batch_size
    batch_size_val = args.batch_size * 2

    # tranformation
    transform = v2.Compose([
        v2.ToTensor() # convert the image to a tensor with values between 0 and 1
    ])

    sailancy_train_dataset = FixationDataset(root_dir="cv2_project_data", image_file="cv2_project_data/train_images.txt", 
						   fixation_file="cv2_project_data/train_fixations.txt", image_transform=transform, fixation_transform=transform)
    
    sailancy_val_dataset = FixationDataset(root_dir="cv2_project_data", image_file="cv2_project_data/val_images.txt", 
						   fixation_file="cv2_project_data/val_fixations.txt", image_transform=transform, fixation_transform=transform)

    # image and fixation are the same for the test dataset as there is no ground truth (fixation) for the test dataset
    sailancy_test_dataset = FixationDataset(root_dir="cv2_project_data", image_file="cv2_project_data/test_images.txt", 
						   fixation_file="cv2_project_data/test_images.txt", image_transform=transform, fixation_transform=transform)

    data_loader_train = torch.utils.data.DataLoader(sailancy_train_dataset,
											batch_size=batch_size_train,
											shuffle=True)

    
    data_loader_val = torch.utils.data.DataLoader(sailancy_val_dataset,
											batch_size=batch_size_val,
											shuffle=False)
    
    data_loader_test = torch.utils.data.DataLoader(sailancy_test_dataset, 
                                            batch_size=1, 
                                            shuffle=False)
    
    one_batch = next(iter(data_loader_train)) # get one batch for testing if the model is working

    # train the model
    import time
    start_time = time.time()
    train(epochs, model, loss_fcn, optimizer, one_batch, data_loader_val, device, args)
    end_time = time.time()

    # evaluate the model
    evaluate(model, data_loader_test, device, args)

    print(f"Runtime: {end_time - start_time} seconds")