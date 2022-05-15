import torch
from swint.swin_transformer import SwinTransformer


def swin(pretrained=False, **kwargs):
    model = SwinTransformer(**kwargs)
    if pretrained:
        pretrained_dict = torch.load('weights/swin_tiny_patch4_window7_224.pth', map_location="cpu")['model']
        model_dict = model.state_dict()
        # 重新制作预训练的权重，主要是减去参数不匹配的层，楼主这边层名为“fc”
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'head' not in k)}
        # 更新权重
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        # model.load_state_dict(torch.load('./models/swin_base_patch4_window12_384.pth', map_location="cpu")['model'])

        # model_state = torch.load('./models/swin_tiny_c24_patch4_window8_256.pth')
        # model.load_state_dict(model_state)

    return model


if __name__ == '__main__':
    swin()
