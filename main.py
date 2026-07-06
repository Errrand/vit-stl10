import torch
import torch.nn as nn
import torch.optim as optim

import sys
from pathlib import Path
root=Path(__file__).resolve().parent
if str(root) not in sys.path:
    sys.path.insert(0,str(root))

from datasets.data import get_dataloaders
from model.vit_tiny import Vit_Tiny
from engine.train import train_one_epoch,evaluate

def main():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader,val_loader,test_loader=get_dataloaders()
    model=Vit_Tiny().to(device)
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.05
    )
    epochs=200
    for epoch in range(epochs):
        train_loss,train_acc=train_one_epoch(model,train_loader,criterion,optimizer,device)
        val_loss,val_acc=evaluate(model,val_loader,criterion,device)
        print(
            f"Epoch: {epoch+1} "
            f"Train Loss:{train_loss:.4f},Train Acc:{train_acc:.4f} "
            f"Eval Loss:{val_loss:.4f},Eval Acc:{val_acc:.4f}"
        )
    test_loss,test_acc=evaluate(model,test_loader,criterion,device)
    print(f"Test Loss:{test_loss:.4f},Test Acc:{test_acc:.4f}")
if __name__=="__main__":
    main()