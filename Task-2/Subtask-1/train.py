import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.ops.focal_loss import sigmoid_focal_loss as SigmoidFocalLoss
import numpy as np
import matplotlib.pyplot as plt
import random

import keyboard
import time
import tqdm

from models import device, ImageSet, ClassicUNet, AttentionUNet, ResNetUNet, ResNetAttendedUNet, SegFormerWrapper

torch.backends.cudnn.benchmark = True # Selecting best possible algorithms

# Setting a manual seed for reproducability (useful for ablations)
torch.manual_seed(67)

def seed_worker(worker_id): # Manual seeding init function for dataloaders
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
g = torch.Generator()
g.manual_seed(42)


def main():
    bce_loss = nn.BCEWithLogitsLoss() # BCE Loss

    def dice_loss(pred, target): # Dice Loss
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(1,2,3))
        denom = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
        dice = (2*intersection + 1e-8) / (denom + 1e-8)
        return 1 - dice.mean()

    focal_loss = SigmoidFocalLoss # Focal Loss

    def CombinedLoss1(pred, target): # BCE + Dice
         return bce_loss(pred, target) + dice_loss(pred, target)

    def CombinedLoss2(pred, target): # BCE + Dice + Focal
             return bce_loss(pred, target) + dice_loss(pred, target) + focal_loss(pred, target, reduction="mean")

    def train(model, tr_dataloader, val_dataloader, epochs, l_rate, patience=10, weight_decay=1e-3, model_name='NULL', clip_grad=False, scheduled_lr=None):
        model.to(device)
        start_time = time.perf_counter()
        loss_fn = CombinedLoss2 
        optimizer = torch.optim.AdamW(model.parameters(), lr=l_rate, betas=(0.9,0.99), weight_decay=weight_decay)
        scaler = torch.amp.GradScaler()
        if scheduled_lr: scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduled_lr[0], gamma=scheduled_lr[1])
        losses = []
        val_losses = []
        best_loss = float('inf') # A large dummy amount, just for initialisation
        pat_count = 0
        epoch=0

        while epoch<epochs: # Gradient descent over epochs
            cancel = False
            total_loss=0
            val_loss=0
            model.train()
            i=0
            # with torch.autograd.set_detect_anomaly(True):
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                for sat, mask in tqdm.tqdm(tr_dataloader, total=len(tr_dataloader)): # Retrieving features for training
                    # start_time = time.perf_counter() #
                    if keyboard.is_pressed('alt+c') and epochs>1:   # A training "cancel" function if epochs seem unfavourable
                                    cancel=True
                                    break
                    sat = sat.to(device, non_blocking=True)
                    mask = mask.to(device, non_blocking=True)
                    pred = model.forward(sat)
                    # print("sanity check")
                    # print(time.perf_counter() - start_time) #
                    if loss_fn==focal_loss:
                        loss = loss_fn(pred, mask, reduction="mean")
                    else:
                        loss = loss_fn(pred, mask)
                    total_loss += loss.item()
                    # start_time = time.perf_counter() #
                    scaler.scale(loss).backward()  # Beginning backprop 
                    if clip_grad:
                        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                    # print(time.perf_counter() - start_time) #
                    i+=1
                    del sat
                    del mask
                # torch.cuda.empty_cache()
            total_loss/=i+1e-8
            if scheduled_lr: scheduler.step()

            model.eval()
            i=0
            with torch.no_grad(), torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                for sat, mask in tqdm.tqdm(val_dataloader, total=len(val_dataloader)):
                    if (keyboard.is_pressed('alt+c') and epochs>1):   # A training "cancel" function if epochs seem unfavourable
                        cancel=True
                        break
                    elif cancel==True:
                        break

                    sat = sat.to(device, non_blocking=True)
                    mask = mask.to(device, non_blocking=True)
                    
                    pred = model.forward(sat)
                    if loss_fn==focal_loss:
                        loss = loss_fn(pred, mask, reduction="mean")
                    else:
                        loss = loss_fn(pred, mask)
                    val_loss+=loss.item()

                    i+=1
                    del sat
                    del mask
                    torch.cuda.empty_cache()
            val_loss/=i+1e-8

            if cancel:
                break

            if val_losses!=[] and val_loss>=best_loss: # Patience mechanism
                if pat_count==patience:
                    break
                else:
                    pat_count+=1
            else:
                pat_count=0

            losses.append(total_loss)
            val_losses.append(val_loss)

            if val_loss<best_loss:
                best_loss = val_loss

                torch.save(model.state_dict(), f'.\\Saved Models\\{model_name if not manual else manual}.pth')

            elapsed_time = time.perf_counter() - start_time
            print(f"(Time Elapsed:{elapsed_time//60:.0f}:{elapsed_time%60:.0f})\tEpoch {epoch+1}, Loss: {total_loss:.4f},  Val Loss: {val_loss:.4f}")
            epoch+=1
        return losses, val_losses



    #  Main

    # Select model here: "classicunet", "resnetunet", "attendedunet", "resnetattendedunet", "deeplabv3", "segformer", "scratchsegformer"
    selected_model = "attendedunet"
    # Put a manual name for the model (and graph)
    manual = ""

    # Setting up dataloaders
    print("Initialising dataloaders...", end='')
    batch_size = 2


    train_loader = DataLoader(ImageSet('train', augment=True), batch_size=batch_size, shuffle=True, num_workers=2,  worker_init_fn=seed_worker, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(ImageSet('val'), batch_size=batch_size, shuffle=True, num_workers=1,  worker_init_fn=seed_worker, pin_memory=True, persistent_workers=True)
    # test_loader = DataLoader(DatasetClass(test_data, tokenizer, sample_data=None), batch_size=batch_size, shuffle=True, num_workers=0)
    print(" Done.\n")

    # Initialisation
    print("Initialising model...", end='')

    if selected_model == "classicunet":
        model = ClassicUNet(num_classes=2)
        model_name = "classicunet_model"
    elif selected_model == "resnetunet":
        model_name = "resnetunet_model"
        model = ResNetUNet(num_classes=2)
    elif selected_model == "attendedunet":
        model = AttentionUNet(num_classes=2)
        model_name = "attendedunet_model"
    elif selected_model == "resnetattendedunet":
        model = ResNetAttendedUNet(num_classes=2)
        model_name = "resnetattendedunet_model"
    elif selected_model == "segformer":
        model = SegFormerWrapper() 
        model_name = "segformer_model"
    elif selected_model == "scratchsegformer":
        model = SegFormerWrapper(pretrained=False) 
        model_name = "scratchsegformer_model"
    print(" Done.\n")

    model.to(device)

    # Training
    print("Training model... (Press ALT+C to interrupt training)")
    loss_history, val_history = train(model, train_loader, val_loader, epochs=50, l_rate=2e-5, 
                                    patience=6, weight_decay=1e-3, model_name=model_name,
                                    clip_grad=True, scheduled_lr=(10, 0.8))#


    # Recording training loss / validation loss graph
    plt.plot(np.arange(1, len(loss_history)+1),loss_history, color='b', label='training loss')
    plt.plot(np.arange(1, len(val_history)+1),val_history, color='r', label='validation loss')
    plt.ylim(0, np.percentile(loss_history + val_history, 95))
    plt.xlabel('Epochs')
    plt.ylabel('Loss History')
    plt.legend()
    plt.savefig(f'.\\Graphs and Images\\{model_name if not manual else manual}_loss_.png')

if __name__=='__main__':
    main()