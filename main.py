import torch
import torch.nn as nn
import torch.optim as optim
import yaml

import sys
from pathlib import Path
root=Path(__file__).resolve().parent
if str(root) not in sys.path:
    sys.path.insert(0,str(root))

from datasets.data import get_dataloaders
from model.vit_tiny import Vit_Tiny
from engine.train import train_one_epoch,evaluate

def main():
    save_dir=root/"outputs"
    save_dir.mkdir(parents=True,exist_ok=True) #创建文件夹，如果父目录不存在也一起创建，如果目录已经存在不报错
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader,val_loader,test_loader=get_dataloaders()
    model=Vit_Tiny().to(device)
    criterion=nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer=optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.05
    )
    epochs=200
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs, #一个完整余弦下降周期有多少步
        eta_min=1e-6 #学习率从初始3e-4下降到1e-6
    )
    best_acc=0.0
    history=[] #list[dict]类型
    for epoch in range(epochs):
        train_loss,train_acc=train_one_epoch(model,train_loader,criterion,optimizer,scheduler,device)
        val_loss,val_acc=evaluate(model,val_loader,criterion,device)
        print(
            f"Epoch: {epoch+1} "
            f"Train Loss:{train_loss:.4f},Train Acc:{train_acc:.4f} "
            f"Eval Loss:{val_loss:.4f},Eval Acc:{val_acc:.4f}"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_dir/"best_model.pth")
        history.append({
            "epoch":epoch+1,
            "train_loss":float(train_loss),
            "train_acc":float(train_acc),
            "val_loss":float(val_loss),
            "val_acc":float(val_acc)
        })
        torch.save({
            "epoch":epoch+1,
            "model":model.state_dict(),
            "optimizer":optimizer.state_dict(),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss":val_loss,
            "val_acc":val_acc
        },"outputs/checkpoint.pth")
        info={
            "epoch":epoch+1,
            "train_loss":train_loss,
            "train_acc":train_acc,
            "val_loss":val_loss,
            "val_acc":val_acc
        }
        #把当前训练的关键指标和配置信息保存成yaml文件，用于记录实验状态和复现实验
        with open(save_dir/"checkpoint.yaml","w",encoding="utf-8") as f:
            yaml.dump(info,f,sort_keys=False)
    with open(save_dir/"history.yaml","w",encoding="utf-8") as f:
        yaml.safe_dump({"history":history},f,sort_keys=False) #要保存的数据,文件对象和保持你字典原来key的顺序
    test_loss,test_acc=evaluate(model,test_loader,criterion,device)
    print(f"Test Loss:{test_loss:.4f},Test Acc:{test_acc:.4f}")
if __name__=="__main__":
    main()
