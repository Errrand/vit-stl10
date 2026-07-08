import json
import yaml

#控制python运行系统的工具箱
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
root=Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0,str(root))




def evaluate(model,data_loader,criterion,device):
    model.eval()
    total_loss=0.0
    correct=0
    total=0
    with torch.no_grad():
        for images,labels in data_loader:
            images,labels=images.to(device),labels.to(device)
            outputs=model(images)
            loss=criterion(outputs,labels)
            total_loss+=loss.item()*images.size(0)
            preds=outputs.argmax(dim=1)
            correct+=(preds==labels).sum().item()
            total+=labels.size(0)
    return total_loss/total,correct/total

def train_one_epoch(model,data_loader,criterion,optimizer,scheduler,device):
    model.train()
    total_loss=0.0
    correct=0
    total=0
    for images,labels in data_loader:
        images,labels=images.to(device),labels.to(device)
        optimizer.zero_grad()
        outputs=model(images)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss+=loss.item()*images.size(0)
        preds=outputs.argmax(dim=1)
        correct+=(preds==labels).sum().item()
        total+=labels.size(0)
    return total_loss/total,correct/total