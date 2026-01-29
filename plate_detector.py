# plate_detector.py
import os
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import sys

def resource_path(relative_path):
    """获取资源的绝对路径，用于PyInstaller打包后访问资源文件"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class ImprovedPlateDetector(nn.Module):
    def __init__(self):
        super(ImprovedPlateDetector, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.regressor = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.regressor(x)
        return x

# 在 plate_detector.py 中修改 detect_plate 函数
def detect_plate(image_path, output_path):
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedPlateDetector()
    
    # 使用resource_path获取模型路径
    model_path = resource_path('models/plate_detector_best.pth')
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"车牌检测模型文件不存在: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    model.eval()


    # 模型训练时的固定输入尺寸
    model_input_size = (250, 400)  # (width, height)
    
    # 加载原始图像
    original_image = Image.open(image_path).convert('RGB')
    original_size = original_image.size  # (原始宽度, 原始高度)
    
    # 将输入图像resize到模型需要的尺寸 (250, 400)
    resized_image = original_image.resize(model_input_size, Image.Resampling.LANCZOS)
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 推理 - 在resize后的图像上进行预测
    input_tensor = transform(resized_image).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_output = model(input_tensor)
    
    # 后处理 - 将预测坐标转换回原始图像坐标
    pred_bbox_normalized = pred_output[0].cpu().numpy()
    
    # 1. 首先在resize后的图像上得到像素坐标
    pred_x1_resized = int(pred_bbox_normalized[0] * model_input_size[0])
    pred_y1_resized = int(pred_bbox_normalized[1] * model_input_size[1])
    pred_x2_resized = int(pred_bbox_normalized[2] * model_input_size[0])
    pred_y2_resized = int(pred_bbox_normalized[3] * model_input_size[1])
    
    # 2. 计算缩放比例
    scale_x = original_size[0] / model_input_size[0]  # 原始宽度 / 模型输入宽度
    scale_y = original_size[1] / model_input_size[1]  # 原始高度 / 模型输入高度
    
    # 3. 将坐标等比例转换回原始图像尺寸
    orig_x1 = int(pred_x1_resized * scale_x)
    orig_y1 = int(pred_y1_resized * scale_y)
    orig_x2 = int(pred_x2_resized * scale_x)
    orig_y2 = int(pred_y2_resized * scale_y)
    
    # 确保坐标在原始图像范围内
    orig_x1 = max(0, min(orig_x1, original_size[0] - 1))
    orig_y1 = max(0, min(orig_y1, original_size[1] - 1))
    orig_x2 = max(orig_x1 + 1, min(orig_x2, original_size[0]))
    orig_y2 = max(orig_y1 + 1, min(orig_y2, original_size[1]))
    
    # 裁剪并保存车牌区域
    plate_region = original_image.crop((orig_x1, orig_y1, orig_x2, orig_y2))
    plate_region.save(output_path)
    
    return [orig_x1, orig_y1, orig_x2, orig_y2]

# 使用示例
if __name__ == "__main__":
    # 输入图像路径
    image_path = 'input_images/1.jpg'
    
    # 输入保存路径
    output_path = 'output_results/1.jpg'
    
    # 检测并保存
    bbox = detect_plate(image_path, output_path)
    
    print(f"检测完成!")
    print(f"车牌位置: {bbox}")
    print(f"车牌图片已保存到: {os.path.abspath(output_path)}")