from torch import nn
import ai8x

class B2RGB(nn.Module):

    def __init__(self,num_classes=None,  # pylint: disable=unused-argument
            num_channels=1,
            dimensions=(64, 64),  # pylint: disable=unused-argument
            bias=True,
            **kwargs):
        super().__init__()

        self.conv1 = ai8x.FusedConv2dReLU(3, 16, 3, padding=1, **kwargs)
        self.conv1_2 = ai8x.FusedConv2dReLU(16, 32, 3, padding=1, **kwargs)
        self.conv1_3 = ai8x.FusedConv2dReLU(32, 64, 3, padding=1, **kwargs)
        self.conv1_4 = ai8x.FusedConv2dReLU(64, 128, 3, padding=1, **kwargs)
        self.conv1_5 = ai8x.FusedConv2dReLU(128, 128, 3, padding=1, **kwargs)
        self.conv2 = ai8x.FusedConv2dReLU(128, 64, 1, padding=0, **kwargs)
        self.conv3 = ai8x.FusedConv2dReLU(64, 32, 3, padding=1, **kwargs)
        self.conv3_2 = ai8x.Conv2d(32, 3, 3, padding=1, **kwargs)
        
    def forward(self, x): 
        
        x = self.conv1(x)
        x = self.conv1_2(x)
        x = self.conv1_3(x)
        x = self.conv1_4(x)
        x = self.conv1_5(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv3_2(x)
        
        return x


def b2rgb(pretrained=False, **kwargs):
    assert not pretrained
    return B2RGB(**kwargs)


models = [
    {
        'name': 'b2rgb',
        'min_input': 1,
        'dim': 2,
    },
]
