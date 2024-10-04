import torch
from torch import nn,optim,autograd
from torchvision import datasets ,transforms
from torch.utils.data import DataLoader

class condition_BatchNorm2d(nn.Module):
    def __init__(self,num_features,num_classes): #num_features 通道数   num_classes 类别数
        super(condition_BatchNorm2d,self).__init__()

        self.num_features=num_features
        self.bn=nn.BatchNorm2d(num_features,affine=False)
        self.embed=nn.Embedding(num_classes,num_features*2)
        self.embed.weight.data[:,:num_features].normal_(1,0.02)
        self.embed.weight.data[:,num_features:].zero_()

    def forward(self,x,y):
        out=self.bn(x)
        gamma,beta=self.embed(y).chunk(2,1)  #在dim=1上一分为二， [num_classes,num_features*2]==>[num_classes,num_features]
        gamma=gamma.view(-1,self.num_features,1,1)
        beta=beta.view(-1,self.num_features,1,1)
        out=gamma*out+beta  #[b,ch,1,1]*[b,ch,x,x]==>[b,ch,x,x]  广播
        return out


class ResBlk(nn.Module):
    def __init__(self,ch_in,ch_out,stride=1,num_classes=0):
        super(ResBlk,self).__init__()

        self.bn1 = condition_BatchNorm2d(num_features=ch_in, num_classes=num_classes)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn2 = condition_BatchNorm2d(num_features=ch_out, num_classes=num_classes)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)

        self.shortcut=nn.Conv2d(ch_in,ch_out,stride=stride,kernel_size=1,padding=0)

    def forward(self,x,labels):
        labels=labels.long().view(-1)
        out = self.bn1(x, labels)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.bn2(out, labels)
        out = self.relu2(out)
        out = self.conv2(out)
        output=out+self.shortcut(x)
        return output

#test
# test=torch.randn(32,4,128,128)
# labels = torch.randint(0, 10, (32,))
# print(labels.size())
# model=ResBlk(4,64,stride=2,num_classes=10)
# print(model(test,labels).shape)


def truncated_noise_sample(batch_size, z_dim, truncation=0.5): #截断函数
    """
    生成截断的潜在向量。

    参数:
    - batch_size: 批量大小
    - z_dim: 潜在向量的维度
    - truncation: 截断阈值

    返回:
    - 截断的潜在向量
    """
    noise=torch.randn(batch_size,z_dim)
    truncated_noise = torch.clamp(noise, -truncation, truncation)  #将noise限制在[-0.5,0.5]
    return truncated_noise

#test
# batch_size = 16
# z_dim = 128
# truncation = 0.5
# truncated_z = truncated_noise_sample(batch_size, z_dim, truncation)
# print('截断的潜在向量 :',truncated_z.size())

class Generator(nn.Module):  #在使用生成器前先调用截断函数
    def __init__(self,z_dim,g_dim,image_size,num_classes):
        super(Generator,self).__init__()

        self.z_dim=z_dim
        self.g_dim=g_dim
        self.image_size=image_size
        self.init_size=image_size//2

        self.linear_1=nn.Sequential(
            nn.Linear(z_dim , g_dim * 8 * self.init_size**2)
        )

        self.res_blocks=nn.Sequential(
            ResBlk(g_dim * 8, g_dim* 8 ,stride=1,num_classes=num_classes),
            ResBlk(g_dim * 8 ,g_dim * 4,stride=1,num_classes=num_classes),
            ResBlk(g_dim * 4, g_dim * 2,stride=1,num_classes=num_classes),
            ResBlk(g_dim * 2, g_dim,stride=1,num_classes=num_classes)
        )

        self.up_sample_1=nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(g_dim, g_dim, kernel_size=3, stride=1, padding=1)
        )
        self.up_sample_2=condition_BatchNorm2d(num_classes=num_classes,num_features=g_dim)
        self.relu=nn.ReLU(True)

        # self.up_sample=nn.Sequential(
        #     nn.Upsample(scale_factor=2),
        #     nn.Conv2d(g_dim,g_dim,kernel_size=3,stride=1,padding=1),
        #     condition_BatchNorm2d(num_classes=num_classes,num_features=g_dim),
        #     nn.ReLU(True)
        # )

        self.final_layer=nn.Sequential(
            nn.Conv2d(g_dim ,3,kernel_size=3,stride=1,padding=1),
            nn.Tanh()
        )

    def forward(self,z,labels):
        #print(z.size())
        out=self.linear_1(z)
        out=out.view(-1, self.g_dim * 8 , self.init_size, self.init_size)
        for block in self.res_blocks:
            out = block(out, labels)
        out=self.up_sample_1(out)
        out=self.up_sample_2(out,labels)
        out=self.relu(out)
        out=self.final_layer(out)
        return out

#test
# z_dim = 100
# g_dim = 64
# image_size = 128
# batch_size=12
# truncation = 0.5
# truncated_z = truncated_noise_sample(batch_size, z_dim, truncation)
# labels = torch.randint(0, 10, (12,))
# G = Generator(z_dim, g_dim, image_size,num_classes=10)
# print(G(truncated_z,labels).size())

class Discriminator(nn.Module):
    def __init__(self,d_dim,image_size):
        super(Discriminator,self).__init__()

        self.d_dim=d_dim
        self.image_size=image_size

        self.conv=nn.Sequential(
            nn.Conv2d(3,d_dim,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(d_dim),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(d_dim,d_dim*2,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(d_dim*2),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(d_dim*2,d_dim*4,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(d_dim*4),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(d_dim*4,d_dim*8,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(d_dim*8),
            nn.LeakyReLU(0.2,inplace=True)
        )

        self.down_sample=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  #[b,ch,x,x]==>[b,ch,1,1] 全局平均池化层
            nn.BatchNorm2d(d_dim*8),
            nn.LeakyReLU(0.2,inplace=True)
        )

        self.final_layer=nn.Sequential(
            nn.Conv2d(d_dim*8,1,kernel_size=4,stride=1,padding=0),
        )


        self.linear=nn.Sequential(
            nn.Linear(d_dim*8,1),
        )

    def forward(self,x):
        out=self.conv(x)
        #print(out.size())
        out=self.down_sample(out)
        #print(out.size())
        out=out.view(out.size(0),-1)
        out=self.linear(out)
        out=out.view(-1)
        return out

#test
# test=torch.randn(12,3,224,224)
# image_size = 224
# d_dim = 64
# D = Discriminator(image_size=image_size, d_dim=d_dim)
# print(D(test).size())
def gradient_penalty(D,x_real,x_fake,batch_size):
    #[b,1,1,1]
    t=torch.rand(batch_size,1,1,1,device=x_real.device)
    #[b,3,32,32]
    t=t.expand_as(x_real)

    mid=t*x_real+((1-t)*x_fake)
    mid.requires_grad_()
    pred=D(mid)
    grads=autograd.grad(outputs=pred,inputs=mid,
                        grad_outputs=torch.ones_like(pred),
                        create_graph=True,retain_graph=True,only_inputs=True)[0] #对mid求导
    gradient=grads.view(grads.size(0),-1)
    gp=((gradient.norm(2, dim=1) - 1) ** 2).mean()  #grads.norm(2,dim=1)求l2范数
    return gp


tf=transforms.Compose([
    transforms.Resize((32,32)),
    #transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data=datasets.CIFAR10(root=r'D:\game\pytorch\简单分类问题\data', train=True, transform=tf, download=True)
loader=DataLoader(data,shuffle=True,batch_size=32)

z_dim = 128
g_dim = 32
image_size = 32
num_classes = 10
generator=Generator(z_dim=z_dim,g_dim=g_dim,image_size=image_size,num_classes=num_classes).to(device)
discriminator=Discriminator(d_dim=g_dim,image_size=image_size).to(device)
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=1e-4)  #加入了正则化
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=1e-4)

import matplotlib.pyplot as plt


def show_generated_images(epoch, generator, labels, num_images=5):
    if torch.cuda.is_available():
        z = torch.randn(num_images, 128).cuda()
    else:
        z = torch.randn(num_images, 128)

    # 确保labels的尺寸与z的批量大小一致
    labels = labels[:num_images]

    fake_images = generator(z, labels).cpu().detach()
    fake_images = (fake_images + 1) / 2  # 将图像从 [-1, 1] 转换到 [0, 1]

    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    for i in range(num_images):
        axes[i].imshow(fake_images[i].permute(1, 2, 0).numpy())
        axes[i].axis('off')
    plt.suptitle(f'Epoch {epoch + 1}')
    plt.show()


for epoch in range(1000):
    for x,y in loader:
        x_real,y_real=x.to(device),y.to(device)
        batch_size=x_real.size(0)
        for p in discriminator.parameters():
            p.requires_grad = True

        #训练判别器
        for _ in range(3):
            optimizer_D.zero_grad()
            z=truncated_noise_sample(batch_size=batch_size,z_dim=z_dim,truncation=0.5).to(device)
            fake_x=generator(z,y_real)

            real_loss=-torch.mean(discriminator(x_real))
            fake_loss=torch.mean(discriminator(fake_x.detach()))  #这里不对fake_x进行反向传播

            gp=gradient_penalty(discriminator,x_real=x_real,x_fake=fake_x,batch_size=batch_size)

            d_loss=real_loss+fake_loss+10*gp
            d_loss.backward()
            optimizer_D.step()

        for p in discriminator.parameters():
            p.requires_grad=False
        #训练生成器
        optimizer_G.zero_grad()
        z = truncated_noise_sample(batch_size=batch_size, z_dim=z_dim, truncation=0.5).to(device)
        fake_x = generator(z, y_real)
        g_loss=-torch.mean(discriminator(fake_x))
        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch + 1}/{1000}]  Loss D: {d_loss.item()}, loss G: {g_loss.item()}")
    if (epoch+1)%10==0:
        show_generated_images(epoch+1, generator,labels=y_real)
