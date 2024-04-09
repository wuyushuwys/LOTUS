import torch.nn as nn
import torch.nn.functional as F

class SwitchableLayerNorm(nn.Module):
    def __init__(self, num_features_list):
        super(SwitchableLayerNorm, self).__init__()
        self.layer_norm = nn.ModuleList()
        for feature in num_features_list:
            self.layer_norm.append(nn.LayerNorm(feature))

    def forward(self, input, idx):
        y = self.layer_norm[idx](input)
        return y


class SwitchableBatchNorm1d(nn.Module):
    def __init__(self, num_features_list):
        super(SwitchableBatchNorm1d, self).__init__()
        self.bn = nn.ModuleList()
        for feature in num_features_list:
            self.bn.append(nn.BatchNorm1d(feature))

    def forward(self, input, idx):
        y = self.bn[idx](input)
        return y
    
class SlimmableBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, num_features_list, **kwargs):
        super(SlimmableBatchNorm1d, self).__init__(num_features=max(num_features_list), **kwargs)
        self.features_list = num_features_list

    def forward(self, input, idx):
        num_features = self.features_list[idx]
        running_mean = self.running_mean[:num_features] if not self.training or self.track_running_stats else None
        running_var = self.running_var[:num_features] if not self.training or self.track_running_stats else None
        weight = self.weight[:num_features] if not self.training or self.affine else None
        bias = self.bias[:num_features] if not self.training or self.affine else None
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        
        if self.training:
            bn_training = True
        else:
            bn_training = (running_mean is None) and (running_var is None)
        y = F.batch_norm(input,
                         running_mean=running_mean,
                         running_var=running_var,
                         weight=weight,
                         bias=bias,
                         training=bn_training,
                         momentum=exponential_average_factor,
                         eps=self.eps,
                         )
        return y

class SlimmableLinear(nn.Linear):
    def __init__(self, in_features_list, out_features_list, bias=True):
        super(SlimmableLinear, self).__init__(
            max(in_features_list), max(out_features_list), bias=bias)
        self.in_features_list = in_features_list
        self.out_features_list = out_features_list

    def forward(self, input, idx):
        in_features = self.in_features_list[idx]
        out_features = self.out_features_list[idx]
        weight = self.weight[:out_features, :in_features]
        if self.bias is not None:
            bias = self.bias[:out_features]
        else:
            bias = self.bias
        return F.linear(input, weight, bias)
    

class SlimmableBlock(nn.Module):

    def __init__(self, in_features_list, out_features_list, act=nn.Identity, dropout=0.1) -> None:
        super(SlimmableBlock, self).__init__()
        self.linear = SlimmableLinear(in_features_list=in_features_list,
                                       out_features_list=out_features_list)
        self.norm = SwitchableBatchNorm1d(num_features_list=out_features_list)
        # self.norm = SwitchableLayerNorm(num_features_list=out_features_list)
        # self.norm = SlimmableBatchNorm1d(num_features_list=out_features_list)
        self.activation = act
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

    
    def forward(self, x, idx):
        x = self.linear(x, idx)
        x = self.norm(x, idx)
        x = self.activation(x)
        x = self.dropout(x)
        return x