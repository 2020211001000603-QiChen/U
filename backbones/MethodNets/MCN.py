import torch.nn as nn

class MCNModel(nn.Module):

    def __init__(self, args, backbone):

        super(MCNModel, self).__init__()
        self.backbone = backbone
    
    def forward(self, text, video, audio, mode='train', labels=None, *args, **kwargs):
        # 处理额外的位置参数
        if len(args) >= 1:
            mode = args[0]
        if len(args) >= 2:
            labels = args[1]

        return self.backbone(text, video, audio, mode)
    