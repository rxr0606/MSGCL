import torch
import torch.nn as nn
import torch.nn.functional as F

class features_proj(nn.Module):
    def __init__(self, hid_dim, out_dim, bias=True):
        super(features_proj, self).__init__()
        self.fc = nn.Linear(hid_dim, out_dim, bias=False)
        self.featdrop=nn.Dropout(0.3)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_dim))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=1.414)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, features):
        feature= self.fc(features)
        
        if self.bias is not None:
            feature += self.bias
        features_proj=self.featdrop(feature)
        return F.elu(features_proj)