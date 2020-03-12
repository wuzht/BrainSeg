# full assembly of the sub-parts to form the complete net

from .unet_parts import *
import numpy as np

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_classes = n_classes
        
        self.inc   = inconv(n_channels, 64)
        self.down1 = down(64,  128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        
        self.up1  = up(1024, 256)
        self.up2  = up(512,  128)
        self.up3  = up(256,  64)
        self.up4  = up(128,  64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        layer1 = self.inc(   x ) 
        layer2 = self.down1( layer1 )
        layer3 = self.down2( layer2 )
        layer4 = self.down3( layer3 )
        layer5 = self.down4( layer4 )
        
        layer4_up = self.up1( layer5,    layer4)
        layer3_up = self.up2( layer4_up, layer3)
        layer2_up = self.up3( layer3_up, layer2)
        layer1_up = self.up4( layer2_up, layer1)     
        output    = self.outc(layer1_up)
        return output

class UNet_shallow(nn.Module):
    def __init__(self, n_channels, n_classes):
        # super(UNet_shallow, self).__init__()
        super().__init__()
        self.n_classes = n_classes
        #######################################################################
        # finish your code here 
        # initial the component here
        self.inc   = inconv(n_channels, 64)
        self.down1 = down(64,  128)
        self.down2 = down(128, 128)

        self.up3  = up(256,  64)
        self.up4  = up(128,  64)
        self.outc = outconv(64, n_classes)
        #######################################################################
        

    def forward(self, x):
        #######################################################################
        # finish your code here
        layer1 = self.inc(   x ) 
        layer2 = self.down1( layer1 )
        layer3 = self.down2( layer2 )

        layer2_up = self.up3( layer3, layer2)
        layer1_up = self.up4( layer2_up, layer1)     
        output    = self.outc(layer1_up)
        return output
        #######################################################################
        # After you write your code above, Please remember delete the following pass
        # pass


class UNet_add(nn.Module):
    def __init__(self, n_channels, n_classes):
        # super(UNet_add, self).__init__()
        super().__init__()
        self.n_classes = n_classes
        
        self.inc   = inconv(n_channels, 64)
        self.down1 = down(64,  128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        
        #####################################################################
        # finish this function with self.up1, self.up2, self.up3, self.up4 and self.outc
        # write your code here
        self.up1  = up_add(512, 256)
        self.up2  = up_add(256,  128)
        self.up3  = up_add(128,  64)
        self.up4  = up_add(64,  64)
        self.outc = outconv(64, n_classes)
        #####################################################################


    def forward(self, x):
        layer1 = self.inc(   x )        # 64
        layer2 = self.down1( layer1 )   # 128
        layer3 = self.down2( layer2 )   # 256
        layer4 = self.down3( layer3 )   # 512
        layer5 = self.down4( layer4 )   # 512
        
        layer4_up = self.up1( layer5,    layer4)    # 512 512 -> 256
        layer3_up = self.up2( layer4_up, layer3)    # 256 256 -> 128
        layer2_up = self.up3( layer3_up, layer2)    # 128 128 -> 64
        layer1_up = self.up4( layer2_up, layer1)    #  64  64 -> 64
        output    = self.outc(layer1_up)

        # print('x {}\n1 {} 2 {} 3 {} 4 {} 5 {}\n4_up {} 3_up {} 2_up {} 1_up{} output {}'.format(
        #     x.shape, layer1.shape, layer2.shape, layer3.shape, layer4.shape, layer5.shape,
        #     layer4_up.shape, layer3_up.shape, layer2_up.shape, layer1_up.shape, output.shape
        # ))
        return output