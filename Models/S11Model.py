import torch.nn as nn
import torch.nn.functional as F

class S11Model(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.prep_layer = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU()
    )
    self.layer1a = self._pre_resblock(in_channels=64, out_channels=128)
    self.layer1b = self._resblock(in_channels=128, out_channels=128)
    self.layer2 = self._pre_resblock(in_channels=128, out_channels=256)
    self.layer3a = self._pre_resblock(in_channels=256, out_channels=512)
    self.layer3b = self._resblock(in_channels=512, out_channels=512)
    self.max_pool = nn.MaxPool2d(kernel_size=(4,4))
    self.fcc = nn.Linear(in_features=512, out_features=10, bias=False)

  def _pre_resblock(self, in_channels, out_channels):
    return nn.Sequential(
      nn.Conv2d(in_channels, out_channels, padding=1, kernel_size=(3,3), stride=1, bias=False),
      nn.MaxPool2d(kernel_size=(2,2)),
      nn.BatchNorm2d(out_channels),
      nn.ReLU())
    
  def _resblock(self, in_channels, out_channels): 
    return nn.Sequential(
      nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1, bias=False),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(),
      
      nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1, bias=False),
      nn.BatchNorm2d(out_channels),
      nn.ReLU()
      )
    
  def forward(self, x):
    x = self.prep_layer(x)
    l = self.layer1a(x)
    r = self.layer1b(l)
    x = self.layer2(l + r)
    l = self.layer3a(x)
    r = self.layer3b(l)
    x = self.max_pool(l + r)
    x = self.fcc(x.view(x.size(0), -1))
    x = x.view(-1, 10)
    return F.log_softmax(x, dim=-1)
