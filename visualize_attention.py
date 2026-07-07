import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt 
import torch.nn.functional as F

root=Path(__file__).resolve().parent
if str(root) not in sys.path:
    sys.path.insert(0,str(root))

from datasets.data import get_dataloaders
from model.vit_tiny import Vit_Tiny

STL10_MEAN = torch.tensor([0.4467, 0.4398, 0.4066]).view(3, 1, 1)
STL10_STD = torch.tensor([0.2603, 0.2566, 0.2713]).view(3, 1, 1)


# 将 Normalize 后的图片还原到 0~1，方便 matplotlib 显示
def denormalize(img):
    img = img.cpu() * STL10_STD + STL10_MEAN
    img = img.clamp(0, 1)
    return img

# 在模型forward过程中，通过hook抓取指定Transformer层的attention map
def collect_attention_maps(model,images,layer_indices):
    attn_maps={}
    hooks=[]
    #生成hook函数的函数
    def make_hook(layer_idx):
        #pytorch forward hook固定需要的函数格式
        #module:被hook的模块本身，input：这个模块的输入，output:这个模块的输出
        def hook(module,input,output):
            #attention map通常形状为[B,heads,N,N],如果是四维，认定它是attn_map
            if output.dim()==4:
                attn_maps[layer_idx]=output.detach().cpu() #detach把这个张量从计算图中分离出去,
        return hook
    for idx in layer_indices:
        hook=model.blocks[idx].attn.dropout.register_forward_hook(make_hook(idx)) #forward执行到第idx个block,然后进入到这个block的attention,执行到attn=self.dropout(x),会自动触发hook
        hooks.append(hook) #返回句柄对象，便于后续删除
    model.eval()
    with torch.no_grad(): #让模型做一次前向传播，在forward过程中触发hook
        _=model(images)
    #用完hook后及时移除，避免后续forward重复触发
    for hook in hooks:
        hook.remove()
    return attn_maps

def cls_attention_to_grid(attn,img_size=96,patch_size=8,head="mean"):
    """
    attn:[B,heads,N,N]
    返回:[H,W] attention heatmap
    """
    attn=attn[0] #取batch中第一张图:[heads,N,N]
    if head=="mean":
        #取CLs token对所有patch token 的注意力，并对多个head求平均 [144]
        cls_attn=attn[:,0,1:].mean(dim=0)
    else:
        #只取指定attention head的CLS attention
        cls_attn=attn[head,0,1:]
    grid_size=img_size//patch_size #图片被切了多少行，多少列patch
    cls_attn=cls_attn.reshape(grid_size,grid_size)
    
    cls_attn=cls_attn.unsqueeze(0).unsqueeze(0) #变成[1,1,12,12],因为F.interpolate()输入通常是四维的
    cls_attn=F.interpolate(
        cls_attn,
        size=(img_size,img_size),
        mode="bilinear", #使用双线性插值进行放大，让相邻区域之间平滑过渡
        align_corners=False #让放大结果更稳定自然
    ) #把小的attention map放大成原来大小
    cls_attn=cls_attn.squeeze() #去掉多余维度,[96,96]
    #归一化到0~1,方便作为热力图显示
    cls_attn=cls_attn/(cls_attn.max()+1e-8)
    return cls_attn

def visualize_layer_attention(model,data_loader,device):
    for i,(images,labels) in enumerate(data_loader):
        if i==1:
            break
    images=images.to(device)

    #分别观察浅层、中层、深层attention
    layer_indices=[0,len(model.blocks)//2,len(model.blocks)-1]
    attn_maps=collect_attention_maps(model,images[:1],layer_indices)
    img=denormalize(images[0])
    img_np=img.permute(1,2,0).numpy() 

    plt.figure(figsize=(12,4))
    plt.subplot(1,4,1)
    plt.imshow(img_np)
    plt.title(f"Image Label: {labels[0].item()}")
    plt.axis("off")

    names=["Shallow","Middle","Deep"]
    for i,layer_idx in enumerate(layer_indices):
        heatmap=cls_attention_to_grid(attn_maps[layer_idx]).numpy()
        plt.subplot(1,4,i+2)
        plt.imshow(img_np)
        plt.imshow(heatmap,cmap="jet",alpha=0.45)
        plt.title(f"{names[i]} layer {layer_idx}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

def main():
    device=torch.device("cuda" if torch.cuda.is_available() else  "cpu")
    _,val_loader,_=get_dataloaders()
    model=Vit_Tiny().to(device)
    checkpoint_path=root/"outputs"/"best_model.pth"
    model.load_state_dict(torch.load(checkpoint_path,map_location=device))
    visualize_layer_attention(model,val_loader,device)

if __name__=="__main__":
    main()