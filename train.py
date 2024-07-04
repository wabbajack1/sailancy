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
from PIL import Image

import argparse
from src.model import Eye_Fixation
from src.dataset import FixationDataset
import wandb

from torchvision.utils import make_grid
from torchvision.io import read_image
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import os

from logging import getLogger
import logging

# make a logger
logger = getLogger(__name__)
# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create console handler and set level to INFO
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Add console handler to logger
logger.addHandler(console_handler)

# set seed for reproducibility
def seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_auc(saliency_map, ground_truth, threshold=None):
    """
    Compute the Area Under the ROC Curve (AUC) for a saliency map. Salary maps and ground truth are expected to be in the same scale, i.e. [0, 1].

    Args:
    saliency_map (torch.Tensor): Predicted saliency map.
    ground_truth (torch.Tensor): Ground truth saliency map (continuous values).

    Returns:
    float: AUC score.
    """
    # Flatten the saliency map and ground truth
    print(saliency_map.shape, ground_truth.shape)
    saliency_map_flat = saliency_map.view(-1).cpu().detach().numpy()
    ground_truth_flat = ground_truth.view(-1).cpu().detach().numpy()

    # Compute the threshold for the ground truth
    # if threshold is None:
    #     threshold = ground_truth_flat.mean().item()
    # binary_ground_truth = (ground_truth_flat >= threshold).astype(float)

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(ground_truth_flat, saliency_map_flat)
    auc_score = auc(fpr, tpr)

    return auc_score

def AUC_Judd(saliencyMap, fixationMap, jitter=True, toPlot=False):
    # saliencyMap is the saliency map
    # fixationMap is the human fixation map (binary matrix)
    # jitter=True will add tiny non-zero random constant to all map locations to ensure
    # 		ROC can be calculated robustly (to avoid uniform region)
    # if toPlot=True, displays ROC curve

    # If there are no fixations to predict, return NaN
    if not fixationMap.any():
        print('Error: no fixationMap')
        score = float('nan')
        return score

    # make the saliencyMap the size of the image of fixationMap
    new_size = np.shape(fixationMap)
    if not np.shape(saliencyMap) == np.shape(fixationMap):
        #from scipy.misc import imresize
        new_size = np.shape(fixationMap)
        np.array(Image.fromarray(saliencyMap).resize((new_size[1], new_size[0])))

        #saliencyMap = imresize(saliencyMap, np.shape(fixationMap))

    # jitter saliency maps that come from saliency models that have a lot of zero values.
    # If the saliency map is made with a Gaussian then it does not need to be jittered as
    # the values are varied and there is not a large patch of the same value. In fact
    # jittering breaks the ordering in the small values!
    if jitter:
        # jitter the saliency map slightly to distrupt ties of the same numbers
        saliencyMap = saliencyMap + np.random.random(np.shape(saliencyMap)) / 10 ** 7

    # normalize saliency map
    saliencyMap = (saliencyMap - saliencyMap.min()) \
                  / (saliencyMap.max() - saliencyMap.min())

    if np.isnan(saliencyMap).all():
        print('NaN saliencyMap')
        score = float('nan')
        return score

    S = saliencyMap.flatten()
    F = fixationMap.flatten()

    Sth = S[F > 0]  # sal map values at fixation locations
    Nfixations = len(Sth)
    Npixels = len(S)

    allthreshes = sorted(Sth, reverse=True)  # sort sal map values, to sweep through values
    tp = np.zeros((Nfixations + 2))
    fp = np.zeros((Nfixations + 2))
    tp[0], tp[-1] = 0, 1
    fp[0], fp[-1] = 0, 1

    for i in range(Nfixations):
        thresh = allthreshes[i]
        aboveth = (S >= thresh).sum()  # total number of sal map values above threshold
        tp[i + 1] = float(i + 1) / Nfixations  # ratio sal map values at fixation locations
        # above threshold
        fp[i + 1] = float(aboveth - i) / (Npixels - Nfixations)  # ratio other sal map values
        # above threshold

    score = np.trapz(tp, x=fp)
    allthreshes = np.insert(allthreshes, 0, 0)
    allthreshes = np.append(allthreshes, 1)

    if toPlot:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        ax.matshow(saliencyMap, cmap='gray')
        ax.set_title('SaliencyMap with fixations to be predicted')
        [y, x] = np.nonzero(fixationMap)
        s = np.shape(saliencyMap)
        plt.axis((-.5, s[1] - .5, s[0] - .5, -.5))
        plt.plot(x, y, 'ro')

        ax = fig.add_subplot(1, 2, 2)
        plt.plot(fp, tp, '.b-')
        ax.set_title('Area under ROC curve: ' + str(score))
        plt.axis((0, 1, 0, 1))
        plt.show()

    return score



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
        for img in tqdm(data_loader_test):
            xb = img["image"].to(device)

            # model prediction
            pred = model(xb)
            predictions = torch.max(pred, dim=0, keepdim=True)[0] # across the batch dimension

            # normalize the predictions
            predictions = predictions - predictions.min()
            max_pred = predictions / predictions.max() * 255.0

            max_pred = max_pred.byte()
            predictions_list.append(max_pred) # to sigmoid
    
    # Get the list of image filenames from the dataset
    image_file_names = data_loader_test.dataset.image_files
    predictions_list = torch.cat(predictions_list, dim=0)
    
    # Iterate through the predictions and save them with the appropriate filename
    for step, pred in enumerate(tqdm(predictions_list)):
        # Extract the base name from the original image file name (without extension)
        base_name = os.path.splitext(os.path.basename(image_file_names[step]))[0]
        
        # Construct the new filename using the base name
        new_filename = f"/export/scratch/9erekmen/predictions/prediction-{base_name}.png"
        
        # Convert the prediction to a numpy array and save it as an image
        imageio.imwrite(new_filename, pred.squeeze(0).cpu().numpy()) # pred (1, 1, H, W)

        # Log the predictions to wandb
        pred = pred.float() # convert to float for visualization
        grid_pred = make_grid(pred.unsqueeze(0), nrow=1) # make grid out of one image
        wandb.log({"eval/images": [wandb.Image(grid_pred, caption=new_filename)]}, step=step) # log the image to wandb

    

def train(epochs, model, loss_fcn, optimizer, data_loader_train, data_loader_val, device, args, logger):
     # training loop
    
    step = 0
    for epoch in range(epochs):
        model.train()
        print(f"Epoch: {epoch+1}")

        # for _ in range(100):
        for _, (img) in enumerate(tqdm(data_loader_train)): # over the sampling dimension
            step += 1
            xb = img["image"].to(device) 
            yb = img["fixation"].to(device)

            # model prediction
            pred = model(xb)

            # loss
            loss = loss_fcn(pred, yb)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # compute accuracy (AUC)
            # auc_score = compute_auc(torch.sigmoid(pred), yb)
            # auc_score = AUC_Judd(pred.cpu().detach(), yb.cpu().detach(), jitter=True, toPlot=False)
            # print(auc_score)

            if (step+1) % args.log_steps == 0:
                print("Step:", step, "Loss:", loss.item())
                
                # log in wandb
                grid_fix = make_grid(img["fixation"][:8], nrow=4)
                grid_raw = make_grid(img["raw_image"][:8], nrow=4)
                grid_prediction = make_grid(torch.sigmoid(pred)[:8], nrow=4)
                wandb.log({"training/loss": loss.item(),
                            "training/images/tgt_fixation": [wandb.Image(grid_fix)],
                            "training/images/raw_image": [wandb.Image(grid_raw)],
                            "training/images/prediction": [wandb.Image(grid_prediction)]}, step=step)
        
        correct = 0
        total = 0
        loss_val = 0
        # evaluate after each epoch
        logger.info("Validation of the model.")
        with torch.no_grad():
            for img in tqdm(data_loader_val):
                xb = img["image"].to(device)
                yb = img["fixation"].to(device)

                # model prediction
                pred = model(xb)
                loss_val += loss_fcn(pred, yb)

                predictions = torch.max(pred, dim=0)[0]

                # compute accuracy (AUC)
                # auc_score = compute_auc(torch.sigmoid(pred), yb)
                # auc_score = AUC_Judd(pred.cpu().detach(), yb.cpu().detach(), jitter=True, toPlot=False)

                # Update counters
                total += yb.size(0)
                correct += (predictions == yb).sum().item()


        # Calculate accuracy
        accuracy = correct / total
        print(f"Accuracy: {accuracy:.4f}, Loss: {loss_val/total:.4f}")
        wandb.log({"validation/accuracy": accuracy, "validation/loss": loss_val/total, "validation/auc": 0, "epoch": epoch+1}, step=step)

        print(f"Saving model after epoch {epoch}.")
        # save model after each epoch
        torch.save(
            {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr': optimizer.param_groups[0]['lr'],
            }, f"/export/scratch/9erekmen/sailancy_model/sailancy_model_epoch_{epoch}.pt"
        )


if __name__ == "__main__":
    # create directory to save model
    os.makedirs("/export/scratch/9erekmen/sailancy_model", exist_ok=True)
    os.makedirs("/export/scratch/9erekmen/predictions", exist_ok=True)

    # parse args
    parser = argparse.ArgumentParser(
                    prog='sailancy paramter training.',
                    description='Train sailancy model.',
                    epilog='Choose device.')
    
    parser.add_argument('--device', type=str, default=False,
                    help='Choose device: cuda | mps | cpu.')
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
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for dataloader")
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
        "seed": args.seed,
        "num_workers": args.num_workers
        },

        mode = "online" if args.log else "disabled"
    )

    
    # choose backend between mps or cuda
    if args.device == "mps":
        device = "mps" if (torch.backends.mps.is_available() and args.device) else "cpu"
    elif args.device == "cuda":
        device = "cuda" if (torch.cuda.is_available() and args.device) else "cpu"
    else:
        device = "cpu"

    print("Model backend:", device)


    # load model, optimizer and state dict
    if args.resume_training is not None: 
        print(f"Resuming training from epoch: {args.resume_training}; Loading Model: sailancy_model_epoch_{args.resume_training}")
        checkpoint = torch.load(f"/export/scratch/9erekmen/sailancy_model/sailancy_model_epoch_{args.resume_training}.pt")
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

    sailancy_train_dataset = FixationDataset(root_dir="/export/scratch/CV2", image_file="/export/scratch/CV2/train_images.txt", 
						   fixation_file="/export/scratch/CV2/train_fixations.txt", image_transform=transform, fixation_transform=transform)
    
    sailancy_val_dataset = FixationDataset(root_dir="/export/scratch/CV2", image_file="/export/scratch/CV2/val_images.txt", 
						   fixation_file="/export/scratch/CV2/val_fixations.txt", image_transform=transform, fixation_transform=transform)

    # image and fixation are the same for the test dataset as there is no ground truth (fixation) for the test dataset
    sailancy_test_dataset = FixationDataset(root_dir="/export/scratch/CV2", image_file="/export/scratch/CV2/test_images.txt", 
						   fixation_file="/export/scratch/CV2/test_images.txt", image_transform=transform, fixation_transform=transform)

    
    assert args.num_workers < os.cpu_count(), "Number of workers cannot be more than the number of cores available and should be less than the number of cores available, because of balancing the load (cpu and gpu usage)."
    data_loader_train = torch.utils.data.DataLoader(sailancy_train_dataset,
											batch_size=batch_size_train,
											shuffle=True, num_workers=args.num_workers)

    
    data_loader_val = torch.utils.data.DataLoader(sailancy_val_dataset,
											batch_size=batch_size_val,
											shuffle=False, num_workers=args.num_workers)
    
    data_loader_test = torch.utils.data.DataLoader(sailancy_test_dataset, 
                                            batch_size=1, 
                                            shuffle=False, num_workers=args.num_workers)
    
    one_batch = next(iter(data_loader_train)) # get one batch for testing if the model is working

    # size of dataloaders
    logger.info(f"Size of train dataloader: {len(data_loader_train)} Batches")
    logger.info(f"Size of val dataloader: {len(data_loader_val)} Batches")
    logger.info(f"Size of test dataloader: {len(data_loader_test)} Batches")
    logger.info(f"Number of cores for dataloader: {data_loader_train.num_workers}")

    # train the model
    import time
    logger.info("Training the model.")
    start_time = time.time()
    train(epochs, model, loss_fcn, optimizer, data_loader_train, data_loader_val, device, args, logger)
    end_time = time.time()

    # evaluate the model
    logger.info("Evaluating the model.")
    evaluate(model, data_loader_test, device, args)

    print(f"Runtime: {end_time - start_time} seconds")