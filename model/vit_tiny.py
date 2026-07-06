import torch
import torch.nn as nn

class patchEmbed(nn.Module):
    def __init__(self,img_size=96,patch_size=8,in_chans=3,embed_dim=192):
        super().__init__()
        self.img_size=img_size
        self.patch_size=patch_size
        self.n_patches=(img_size//patch_size)**2
        self.proj=nn.Conv2d( #输出大小 = (输入大小 + 2 × padding - kernel_size) / stride + 1
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
    def forward(self,x):
        # 经过ToTensor,x: [B, 3, 96, 96]
        x=self.proj(x) #[64,192,12,12]
        x=x.flatten(2) #[B,C,N]，B=batch size,N=patch nums,C= token特征维度
        x=x.transpose(1,2) #[B,N,C],[64,144,192]
        return x

#完成多头自注意力机制，让模型用不同的视角(head)去分析同一张图
#输入,dim:每个token的特征维度，heads:注意力头数量,dim_head:每个注意力头维度
class Attention(nn.Module):
    def __init__(self,embed_dim,heads=3,dropout=0.1):
        super().__init__()
        self.heads=heads #注意力头数量
        self.dim_head=embed_dim//heads #每个头的维度

        self.scale=self.dim_head ** -0.5

        #线性映射得到q,k,v
        self.q_proj=nn.Linear(embed_dim,embed_dim)
        self.k_proj=nn.Linear(embed_dim,embed_dim)
        self.v_proj=nn.Linear(embed_dim,embed_dim)
        #最后的输出投影
        self.out_proj=nn.Linear(embed_dim,embed_dim)
        #dropout随机让某些神经元失活，防止过拟合
        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        B,N,D=x.shape #x.shape返回是一个类似元组的对象，python支持解包赋值
        #1.线性映射得到q,k,v
        q=self.q_proj(x)
        k=self.k_proj(x)
        v=self.v_proj(x)

        #2.把[B,N,D]拆成多头
        #[B,N,D]变成[B,heads,N,dim_head],[64,3,144,64]
        q=q.reshape(B,N,self.heads,self.dim_head).permute(0,2,1,3)
        k=k.reshape(B,N,self.heads,self.dim_head).permute(0,2,1,3)
        v=v.reshape(B,N,self.heads,self.dim_head).permute(0,2,1,3)

        #3.计算注意力分数,[B,heads,N,dim_head]*[B,heads,dim_head,N]=[B,heads,N,N]
        #每个query和所有key做点积，得到相似度矩阵，第i个query对第j个key的相似度
        attn=torch.matmul(q,k.transpose(-2,-1))*self.scale

        #4.把注意力分数用softmax得到注意力权重
        #对每个query token,计算它关注所有key的概率分布，所以必须在维上做softmax
        attn=torch.softmax(attn,dim=-1)

        #5.dropout防止过拟合
        attn=self.dropout(attn)

        #6.用注意力权重去加权v,[B,heads,N,N]*[B,heads,N,D]=[B,heads,N,D]
        #attn为每个token对所有token的关注程度，v为每个token的内容信息
        #得到融合上下文信息后的token表示
        out=torch.matmul(attn,v)

        #7.重新拼接多个head结果
        #先转回[B,N,heads,dim_head]
        out=out.permute(0,2,1,3)
        #展平成[B,N,D]
        out=out.reshape(B,N,D)

        #8.最后做一次输出投影
        out=self.out_proj(out)

        #输出特征再随机失活
        return self.dropout(out)

#对每个token的特征进行非线性变换,增强表达能力，同时保持输入输出维度一致，方便残差连接
class MLP(nn.Module):
    def __init__(self,dim,hidden_dim,dropout=0.1):
        super().__init__()
        self.fc1=nn.Linear(dim,hidden_dim)
        self.act1=nn.GELU()
        self.fc2=nn.Linear(hidden_dim,dim)
        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        x=self.fc1(x)
        x=self.act1(x)
        x=self.dropout(x)
        x=self.fc2(x)
        return self.dropout(x)

#Transformer Block
class Block(nn.Module):
    def __init__(self,embed_dim,heads=3,mlp_ratio=4.0,dropout=0.1):
        super().__init__()
        #对输入张量的最后若干个维度做归一化,所以参数必须匹配输入张量的最后几个维度
        self.norm1=nn.LayerNorm(embed_dim)
        self.attn=Attention(embed_dim,heads=heads,dropout=dropout)
        self.norm2=nn.LayerNorm(embed_dim)
        self.mlp=MLP(embed_dim,int(embed_dim*mlp_ratio),dropout=dropout)
    def forward(self,x):
        out=x+self.attn(self.norm1(x))
        out=out+self.mlp(self.norm2(out))
        return out

class Vit_Tiny(nn.Module):
    def __init__(
            self,
            img_size=96,
            patch_size=8,
            in_chans=3,
            embed_dim=192,
            num_classes=10,
            mlp_ratio=4.0,
            dropout=0.1,
            heads=3,
            L=12
    ):
        super().__init__()
        #patch embedding
        self.patch_embed=patchEmbed(img_size=img_size,
                                    patch_size=patch_size,
                                    in_chans=in_chans,
                                    embed_dim=embed_dim)
        self.num_patches=self.patch_embed.n_patches #patches数量
        #嵌入CLS Token,
        #CLS token 的大小是 [B, 1, D]，其中 D 等于 ViT 的 
        # embedding dim；它一开始只是可学习参数，经过多层 self-attention 后，
        # 会聚合整张图片的全局语义信息，最后通常用于分类。
        self.cls_token=nn.Parameter(torch.zeros(1,1,embed_dim))
        #嵌入position embedding
        self.pos_embed=nn.Parameter(torch.zeros(1,self.num_patches+1,embed_dim))
        #dropout防止过拟合
        self.dropout=nn.Dropout(dropout)
        #Encoder Block
        self.blocks=nn.ModuleList([
            Block(embed_dim=embed_dim,heads=heads,mlp_ratio=mlp_ratio,dropout=dropout)
            for _ in range(L)
        ])
        self.norm=nn.LayerNorm(embed_dim)
        self.head=nn.Linear(embed_dim,num_classes)
    def forward(self,x):
        B=x.shape[0]

        #1.patch embedding
        x=self.patch_embed(x)

        #2.加入cls token
        cls_token=self.cls_token.expand(B,-1,-1) #[B,1,D]
        #caa(tensors,dim=0),Tensors是要拼接的张量序列，dim是沿哪个维度拼接
        x=torch.cat([cls_token,x],dim=1) #[B,N+1,D]

        #3.加入position embedding
        x=x+self.pos_embed #[B,N+1,D]

        #4.dropout
        x=self.dropout(x)

        #5.经过多个transformer block
        #ModuleList可以像普通列表一样索引和遍历
        for block in self.blocks:
            x=block(x)

        #6.取CLS token的输出用于分类
        x=self.norm(x[:,0]) #[B,D],x[:, 0] 会少一维
        x=self.head(x) #[B,10]
        return x 
    
