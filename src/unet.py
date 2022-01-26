import torch
import torch.nn as nn
#from torch.nn.modules.activation import ReLU
#from torch.nn.modules.conv import Conv2d
#print('asdfda')
def block(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3),
        nn.ReLU(inplace = True),
        nn.Conv2d(out_c, out_c, kernel_size = 3),
        nn.ReLU(inplace = True),
    )
    return conv
#we actually define a function
def bloc_up(in_c, out_c):
    conv = nn.ConvTranspose2d(in_c, out_c, kernel_size= 2, stride=2)
    conv = block(conv[1], conv[1]*2)

#cropping
def crop_img(tensor, target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2
    return tensor[:, :, delta:(tensor_size - delta), delta:(tensor_size - delta)]



class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.max_pool_2by2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.down_conv1 = block(1, 64)
        self.down_conv2 = block(64, 128)
        self.down_conv3 = block(128, 256)
        self.down_conv4 = block(256, 512)
        self.down_conv5 = block(512, 1024)

        self.transpose1 = nn.ConvTranspose2d(
            in_channels=1024, 
            out_channels=512,
            kernel_size=2,
            stride=2
            )
        self.up_conv1 = block(1024, 512)

        self.transpose2 = nn.ConvTranspose2d(
            in_channels=512, 
            out_channels=256,
            kernel_size=2,
            stride=2
            )
        self.up_conv2 = block(512, 256)

        self.transpose3 = nn.ConvTranspose2d(
            in_channels=256, 
            out_channels=128,
            kernel_size=2,
            stride=2
            )
        self.up_conv3 = block(256, 128)

        self.transpose4 = nn.ConvTranspose2d(
            in_channels=128, 
            out_channels=64,
            kernel_size=2,
            stride=2
            )
        self.up_conv4 = block(128, 64)

        self.out = nn.Conv2d(
            in_channels=64,
            out_channels= 1,
            kernel_size=1
            )
        
    
    def forward(self, image):
        #encoder
        x1 = self.down_conv1(image)
        x2 = self.max_pool_2by2(x1)

        print(x2.size())

        x3 = self.down_conv2(x2)
        x4 = self.max_pool_2by2(x3)

        x5 = self.down_conv3(x4)
        x6 = self.max_pool_2by2(x5)

        x7 = self.down_conv4(x6)
        #print(enc7.size())
        x8 = self.max_pool_2by2(x7)

        x9 = self.down_conv5(x8)


        #decoder

        x = self.transpose1(x9)
        y = crop_img(x7, x)
        x = self.up_conv1(torch.cat(([x, y]), 1))

        x = self.transpose2(x)
        y = crop_img(x5, x)
        x = self.up_conv2(torch.cat(([x, y]), 1))

        x = self.transpose3(x)
        y = crop_img(x3, x)
        x = self.up_conv3(torch.cat(([x, y]), 1))

        x = self.transpose4(x)
        y = crop_img(x1, x)
        x = self.up_conv4(torch.cat(([x, y]), 1))
        
        x = self.out(x)
        #print(x.size())
        return x
        
        
        #print(enc9.size())
        
if __name__ == "__main__":
     image = torch.rand(1, 1, 572, 572)
     model = UNet()
     print(model(image)) 



