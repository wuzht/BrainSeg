# full assembly of the sub-parts to form the complete net

from .unet_parts import *
import numpy as np
import torch.nn.functional as F


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


class DropoutUNet(nn.Module):
    def __init__(self, n_channels, n_classes, model_type='Center1', drop_rate=0.2):
        super(DropoutUNet, self).__init__()
        self.n_classes = n_classes
        self.drop_rate = drop_rate
        self.model_type = model_type
        
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

    def forward(self, x, is_dropout=True):
        if self.model_type == 'All':
            layer1 = self.inc( x )
            layer2 = self.down1( layer1 )
            layer2 = F.dropout(layer2, p=self.drop_rate, training=is_dropout)
            layer3 = self.down2( layer2 )
            layer3 = F.dropout(layer3, p=self.drop_rate, training=is_dropout)
            layer4 = self.down3( layer3 )
            layer4 = F.dropout(layer4, p=self.drop_rate, training=is_dropout)
            layer5 = self.down4( layer4 )
            layer5 = F.dropout(layer5, p=self.drop_rate, training=is_dropout)
            layer4_up = self.up1( layer5,    layer4)
            layer4_up = F.dropout(layer4_up, p=self.drop_rate, training=is_dropout)
            layer3_up = self.up2( layer4_up, layer3)
            layer3_up = F.dropout(layer3_up, p=self.drop_rate, training=is_dropout)
            layer2_up = self.up3( layer3_up, layer2)
            layer2_up = F.dropout(layer2_up, p=self.drop_rate, training=is_dropout)
            layer1_up = self.up4( layer2_up, layer1)
            layer1_up = F.dropout(layer1_up, p=self.drop_rate, training=is_dropout)
            output    = self.outc(layer1_up)
        elif self.model_type == 'Encoder':
            layer1 = self.inc( x )
            layer2 = self.down1( layer1 )
            layer2 = F.dropout(layer2, p=self.drop_rate, training=is_dropout)
            layer3 = self.down2( layer2 )
            layer3 = F.dropout(layer3, p=self.drop_rate, training=is_dropout)
            layer4 = self.down3( layer3 )
            layer4 = F.dropout(layer4, p=self.drop_rate, training=is_dropout)
            layer5 = self.down4( layer4 )
            layer5 = F.dropout(layer5, p=self.drop_rate, training=is_dropout)
            layer4_up = self.up1( layer5,    layer4)
            layer3_up = self.up2( layer4_up, layer3)
            layer2_up = self.up3( layer3_up, layer2)
            layer1_up = self.up4( layer2_up, layer1)
            output    = self.outc(layer1_up)
        elif self.model_type == 'Decoder':
            layer1 = self.inc( x )
            layer2 = self.down1( layer1 )
            layer3 = self.down2( layer2 )
            layer4 = self.down3( layer3 )
            layer5 = self.down4( layer4 )
            layer4_up = self.up1( layer5,    layer4)
            layer4_up = F.dropout(layer4_up, p=self.drop_rate, training=is_dropout)
            layer3_up = self.up2( layer4_up, layer3)
            layer3_up = F.dropout(layer3_up, p=self.drop_rate, training=is_dropout)
            layer2_up = self.up3( layer3_up, layer2)
            layer2_up = F.dropout(layer2_up, p=self.drop_rate, training=is_dropout)
            layer1_up = self.up4( layer2_up, layer1)
            layer1_up = F.dropout(layer1_up, p=self.drop_rate, training=is_dropout)
            output    = self.outc(layer1_up)
        elif self.model_type == 'Center1':
            layer1 = self.inc( x )
            layer2 = self.down1( layer1 )
            layer3 = self.down2( layer2 )
            layer4 = self.down3( layer3 )
            layer5 = self.down4( layer4 )
            layer5 = F.dropout(layer5, p=self.drop_rate, training=is_dropout)
            layer4_up = self.up1( layer5,    layer4)
            layer4_up = F.dropout(layer4_up, p=self.drop_rate, training=is_dropout)
            layer3_up = self.up2( layer4_up, layer3)
            layer2_up = self.up3( layer3_up, layer2)
            layer1_up = self.up4( layer2_up, layer1)
            output    = self.outc(layer1_up)
        elif self.model_type == 'Center2':
            layer1 = self.inc( x )
            layer2 = self.down1( layer1 )
            layer3 = self.down2( layer2 )
            layer4 = self.down3( layer3 )
            layer4 = F.dropout(layer4, p=self.drop_rate, training=is_dropout)
            layer5 = self.down4( layer4 )
            layer5 = F.dropout(layer5, p=self.drop_rate, training=is_dropout)
            layer4_up = self.up1( layer5,    layer4)
            layer4_up = F.dropout(layer4_up, p=self.drop_rate, training=is_dropout)
            layer3_up = self.up2( layer4_up, layer3)
            layer3_up = F.dropout(layer3_up, p=self.drop_rate, training=is_dropout)
            layer2_up = self.up3( layer3_up, layer2)
            layer1_up = self.up4( layer2_up, layer1)
            output    = self.outc(layer1_up)
        elif self.model_type == 'Classifier':
            layer1 = self.inc( x )
            layer2 = self.down1( layer1 )
            layer3 = self.down2( layer2 )
            layer4 = self.down3( layer3 )
            layer5 = self.down4( layer4 )
            layer4_up = self.up1( layer5,    layer4)
            layer3_up = self.up2( layer4_up, layer3)
            layer2_up = self.up3( layer3_up, layer2)
            layer1_up = self.up4( layer2_up, layer1)
            layer1_up = F.dropout(layer1_up, p=self.drop_rate, training=is_dropout)
            output    = self.outc(layer1_up)
        elif self.model_type == 'Mid1':
            layer1 = self.inc( x )
            layer2 = self.down1( layer1 )
            layer3 = self.down2( layer2 )
            layer4 = self.down3( layer3 )
            layer4 = F.dropout(layer4, p=self.drop_rate, training=is_dropout)
            layer5 = self.down4( layer4 )
            layer4_up = self.up1( layer5,    layer4)
            layer3_up = self.up2( layer4_up, layer3)
            layer3_up = F.dropout(layer3_up, p=self.drop_rate, training=is_dropout)
            layer2_up = self.up3( layer3_up, layer2)
            layer1_up = self.up4( layer2_up, layer1)
            output    = self.outc(layer1_up)
        elif self.model_type == 'Mid1-Encoder':
            layer1 = self.inc( x )
            layer2 = self.down1( layer1 )
            layer3 = self.down2( layer2 )
            layer4 = self.down3( layer3 )
            layer4 = F.dropout(layer4, p=self.drop_rate, training=is_dropout)
            layer5 = self.down4( layer4 )
            layer4_up = self.up1( layer5,    layer4)
            layer3_up = self.up2( layer4_up, layer3)
            layer2_up = self.up3( layer3_up, layer2)
            layer1_up = self.up4( layer2_up, layer1)
            output    = self.outc(layer1_up)
        elif self.model_type == 'Mid1-Decoder':
            layer1 = self.inc( x )
            layer2 = self.down1( layer1 )
            layer3 = self.down2( layer2 )
            layer4 = self.down3( layer3 )
            layer5 = self.down4( layer4 )
            layer4_up = self.up1( layer5,    layer4)
            layer3_up = self.up2( layer4_up, layer3)
            layer3_up = F.dropout(layer3_up, p=self.drop_rate, training=is_dropout)
            layer2_up = self.up3( layer3_up, layer2)
            layer1_up = self.up4( layer2_up, layer1)
            output    = self.outc(layer1_up)
        elif self.model_type == 'No':
            layer1 = self.inc( x )
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
        self.inc   = inconv(n_channels, 64)
        self.down1 = down(64,  128)
        self.down2 = down(128, 128)

        self.up3  = up(256,  64)
        self.up4  = up(128,  64)
        self.outc = outconv(64, n_classes)
        #######################################################################
        

    def forward(self, x):
        #######################################################################
        layer1 = self.inc(   x ) 
        layer2 = self.down1( layer1 )
        layer3 = self.down2( layer2 )

        layer2_up = self.up3( layer3, layer2)
        layer1_up = self.up4( layer2_up, layer1)     
        output    = self.outc(layer1_up)
        return output
        #######################################################################


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

        return output