import torch
import torchvision.models as models
import torch.nn as nn

def convert_model(model_path, output_path):
    # 实例化预训练模型
    model = models.mobilenet_v3_small(pretrained=False)  # 预训练设置为 False，我们将加载自己的权重

    # 修改分类器层
    model.classifier = nn.Sequential(
        nn.Linear(in_features=576, out_features=1024, bias=True),
        nn.Hardswish(),
        torch.nn.Dropout(p=0.2, inplace=True), 
        torch.nn.Linear(in_features=1024, out_features=1, bias=True)
    )

    # 加载模型状态字典
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()

    # 转换为 TorchScript
    example_input = torch.rand(1, 3, 384, 384)  # 根据你的模型输入调整大小
    torch.onnx.export(model, example_input, output_path)

if __name__ == '__main__':
    model_path = 'best_model_tl.pth'
    output_path = 'best_model_tl.onnx'
    convert_model(model_path, output_path)