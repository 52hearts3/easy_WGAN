import torch
from torch import nn,optim,autograd
from torch.utils.data import DataLoader
from torchvision import transforms,datasets

class condition_BatchNorm2d(nn.Module):
    def __init__(self,num_ch,num_classes):
        super(condition_BatchNorm2d,self).__init__()

        self.num_ch=num_ch
        self.num_classes=num_classes
        self.bn=nn.BatchNorm2d(num_ch,affine=False)
        self.embed=nn.Embedding(num_classes,num_ch*2)

        self.embed.weight.data[:,:num_ch].normal_(1,0.2)
        self.embed.weight.data[:,num_ch:].zero_()

    def forward(self,x,y):
        x_bn=self.bn(x)
        gamma,beta=self.embed(y).chunk(2,1)
        gamma=gamma.view(-1,self.num_ch,1,1)
        beta=beta.view(-1,self.num_ch,1,1)
        out=gamma*x_bn+beta
        return out

# test=torch.randn(12,3,32,32)
# y=torch.randint(0,10,(12,))
# model=condition_BatchNorm2d(num_ch=3,num_classes=10)
# print(model(test,y).size())

class AttentionBlock(nn.Module):
    def __init__(self,ch_in):
        super(AttentionBlock,self).__init__()

        self.norm=nn.Sequential(
            nn.InstanceNorm2d(ch_in)
        )
        self.ch_in=ch_in
        self.to_qkv=nn.Conv2d(ch_in,ch_in*3,kernel_size=1,stride=1,padding=0)
        self.out_layer=nn.Conv2d(ch_in,ch_in,kernel_size=1,padding=0)

    def forward(self,x):
        b,ch,h,w=x.size()
        x=self.norm(x)
        x_to_qkv=self.to_qkv(x)
        q,k,v=torch.split(x_to_qkv,self.ch_in,dim=1)
        q=q.permute(0,2,3,1).view(b,h*w,ch)
        k=k.view(b,ch,h*w)
        v=v.permute(0,2,3,1).view(b,h*w,ch)
        dot_product=torch.bmm(q,k)*(ch**0.5)
        attention=torch.softmax(dot_product,dim=-1)
        out=torch.bmm(attention,v).view(b,h,w,ch).permute(0,3,1,2)
        output=self.out_layer(out)+x
        return output

test=torch.randn(12,3,32,32)
model=AttentionBlock(ch_in=3)
print(model(test).size())


class ResBlk(nn.Module):
    def __init__(self,ch_in,ch_out,stride=1,num_classes=0):
        super(ResBlk,self).__init__()

        self.bn1=condition_BatchNorm2d(num_ch=ch_in,num_classes=num_classes)
        self.con1=nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=stride,padding=0)
        )
        self.bn2=condition_BatchNorm2d(num_ch=ch_out,num_classes=num_classes)
        self.con2=nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=0)
        )
        self.relu=nn.ReLU()

        self.shortcut=nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch_in,ch_out,stride=stride,padding=0,kernel_size=3)
        )

        self.attention=AttentionBlock(ch_out)

    def forward(self,x,label):
        x_bn1=self.bn1(x,label)
        x_con1=self.con1(x_bn1)
        x_con1=self.relu(x_con1)
        x_bn2=self.bn2(x_con1,label)
        x_con2=self.con2(x_bn2)
        x_con2=self.relu(x_con2)
        x_con2=self.attention(x_con2)
        out=x_con2+self.shortcut(x)
        return out

test=torch.randn(12,3,32,32)
y=torch.randint(0,10,(12,))
model=ResBlk(ch_in=3,ch_out=32,num_classes=10)
print(model(test,y).size())

def truncated_noise_sample(batch_size, z_dim, truncation=0.5): #截断函数
    z=torch.randn(batch_size,z_dim)
    z_=torch.clamp(z,-truncation,truncation)
    return z_

def gradient_penalty(D,x_real,x_fake,batch_size):

    t=torch.rand(batch_size,1,1,1,device=x_real.device)
    t=t.expand_as(x_real)
    mid=t*x_real+(1-t)*x_fake
    mid.requires_grad_(True)
    pred=D(mid)
    grads = autograd.grad(outputs=pred, inputs=mid,
                          grad_outputs=torch.ones_like(pred),
                          create_graph=True, retain_graph=True, only_inputs=True)[0]  # 对mid求导

    gradient=grads.view(grads.size(0),-1)
    gp = ((gradient.norm(2, dim=1) - 1) ** 2).mean()
    return gp

class Generator(nn.Module):
    def __init__(self,z_dim,g_dim,image_size,num_classes):
        super(Generator,self).__init__()

        self.z_dim=z_dim
        self.g_dim=g_dim
        self.image_size = image_size
        self.init_size = image_size // 4

        self.linear=nn.Linear(z_dim,g_dim * 8 * (self.init_size**2))

        self.ResBlk=nn.Sequential(
            ResBlk(g_dim * 8, g_dim * 8, num_classes=num_classes, stride=1),
            ResBlk(g_dim * 8, g_dim * 8, num_classes=num_classes, stride=1),
            ResBlk(g_dim * 8, g_dim * 8, num_classes=num_classes, stride=1),
            ResBlk(g_dim * 8, g_dim * 8, num_classes=num_classes, stride=1)
        )

        self.up_sample_1=nn.Sequential(
            nn.Conv2d(g_dim * 8, g_dim * 16, kernel_size=3,stride=1,padding=1),
            ResBlk(g_dim*16,g_dim*16,num_classes=num_classes),
            ResBlk(g_dim*16,g_dim*16,num_classes=num_classes),
            nn.PixelShuffle(2), #[b,g_dim*16,x,x]==>[b,g_dim*4,2x,2x]
        )
        self.condition_BatchNorm_1=condition_BatchNorm2d(num_ch=g_dim*4,num_classes=num_classes)
        self.up_sample_2=nn.Sequential(
            nn.Conv2d(g_dim*4,g_dim*4,kernel_size=3,stride=1,padding=1),
            ResBlk(g_dim*4,g_dim*4,num_classes=num_classes),
            ResBlk(g_dim * 4, g_dim * 4, num_classes=num_classes),
            nn.PixelShuffle(2) #[b,g_dim*4,2x,2x]==>[b,g_dim,4x,4x]
        )
        self.condition_BatchNorm_2=condition_BatchNorm2d(num_ch=g_dim,num_classes=num_classes)

        self.final_layer = nn.Sequential(
            nn.Conv2d(g_dim, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        self.relu=nn.ReLU()

    def forward(self,x,label):
        x=self.linear(x)
        x=x.view(-1,self.g_dim*8,self.init_size,self.init_size)
        for block in self.ResBlk:
            x=block(x,label)
        for layer in self.up_sample_1:
            if isinstance(layer,ResBlk):
                x=layer(x,label)
            else:
                x=layer(x)
        x=self.condition_BatchNorm_1(x,label)
        x=self.relu(x)
        for layer in self.up_sample_2:
            if isinstance(layer,ResBlk):
                x=layer(x,label)
            else:
                x=layer(x)
        x=self.condition_BatchNorm_2(x,label)
        x=self.relu(x)
        x=self.final_layer(x)
        return x

test=torch.randn(12,32)
label=torch.randint(10,(12,))
z_dim=32
g_dim=8
image_size=32
num_classes=10
model=Generator(z_dim,g_dim,image_size,num_classes)
print(model(test,label).shape)

import torch.nn.utils.spectral_norm as spectral_norm  #引入普归一化

#spectral_norm 是一种正则化技术，用于稳定生成对抗网络（GAN）的训练过程。
# 它通过对每个权重矩阵进行谱归一化，限制其最大奇异值，从而防止梯度爆炸和消失问题。
class Discriminator(nn.Module):
    def __init__(self,d_dim):
        super(Discriminator,self).__init__()

        self.d_dim=d_dim

        self.conv=nn.Sequential(
            spectral_norm(nn.Conv2d(3,d_dim,kernel_size=4,stride=2,padding=1)),
            nn.BatchNorm2d(d_dim),
            nn.LeakyReLU(0.2,inplace=True),
            spectral_norm(nn.Conv2d(d_dim,d_dim*2,kernel_size=4,stride=2,padding=1)),
            nn.BatchNorm2d(d_dim*2),
            nn.LeakyReLU(0.2,inplace=True),
            spectral_norm(nn.Conv2d(d_dim*2,d_dim*4,kernel_size=4,stride=2,padding=1)),
            nn.BatchNorm2d(d_dim*4),
            nn.LeakyReLU(0.2,inplace=True),
            spectral_norm(nn.Conv2d(d_dim*4,d_dim*8,kernel_size=4,stride=2,padding=1)),
            nn.BatchNorm2d(d_dim*8),
            nn.LeakyReLU(0.2,inplace=True)
        )

        self.down_sample=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.BatchNorm2d(d_dim * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )


        self.linear=nn.Linear(d_dim*8,1)

    def forward(self,x):
        x=self.conv(x)
        x=self.down_sample(x)
        x=x.view(x.size(0),-1)
        x=self.linear(x)
        x=x.view(-1)
        return x

test=torch.randn(12,3,32,32)
d_dim=8
model=Discriminator(d_dim)
print(model(test).size())

tf=transforms.Compose([
    # transforms.Lambda(convert_to_rgb),
    transforms.Resize((32,32)),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

data=datasets.CIFAR10(root=r'D:\game\pytorch\简单分类问题\data', train=True, transform=tf, download=True)
loader=DataLoader(data,batch_size=32,shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
z_dim=128
image_size=32
g_dim=32
num_classes=10
generator=Generator(z_dim=z_dim,g_dim=g_dim,image_size=image_size,num_classes=num_classes).to(device)
discriminator=Discriminator(d_dim=g_dim).to(device)
optimizer_G=optim.Adam(generator.parameters(),lr=1e-4,betas=(0.5, 0.999), weight_decay=1e-4)
optimizer_D=optim.Adam(discriminator.parameters(),lr=1e-4,betas=(0.5, 0.999), weight_decay=1e-4)

for epoch in range(1000):
    for x,y in loader:
        x_real,y_real=x.to(device),y.to(device)
        batch_size=x_real.size(0)

        #训练判别器
        for p in discriminator.parameters():
            p.requires_grad=True
        for _ in range(3):
            optimizer_D.zero_grad()
            z=truncated_noise_sample(batch_size=batch_size,z_dim=z_dim,truncation=0.5).to(device)
            fake_x=generator(z,y_real)
            real_loss=-torch.mean(discriminator(x_real))
            fake_loss=torch.mean(discriminator(fake_x.detach()))

            gp = gradient_penalty(discriminator, x_real=x_real, x_fake=fake_x, batch_size=batch_size)

            d_loss = real_loss + fake_loss + 10 * gp
            d_loss.backward()
            optimizer_D.step()

        #训练生成器
        for p in discriminator.parameters():
            p.requires_grad=False

        optimizer_G.zero_grad()
        z = truncated_noise_sample(batch_size=batch_size, z_dim=z_dim, truncation=0.5).to(device)
        fake_x = generator(z, y_real)
        g_loss = -torch.mean(discriminator(fake_x))
        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch + 1}/{1000}]  Loss D: {d_loss.item()}, loss G: {g_loss.item()}")




