import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import cv2
import glob
import time
import sys
from datetime import datetime

def resource_path(relative_path):
    """获取资源的绝对路径，用于PyInstaller打包后访问资源文件"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class PlateRecognizer:
    def __init__(self, hanzi_model_path='hanzi_model.pth', alnum_model_path='alnum_model.pth', 
                 confidence_threshold=0.9, max_retries=10, debug_output_dir='preprocess_debug'):
        
        # 使用resource_path包装模型路径
        hanzi_model_path = resource_path(hanzi_model_path)
        alnum_model_path = resource_path(alnum_model_path)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        self.max_retries = max_retries
        self.debug_output_dir = debug_output_dir
        os.makedirs(self.debug_output_dir, exist_ok=True)
        
        if not os.path.exists(hanzi_model_path):
            raise FileNotFoundError(f"汉字模型文件不存在: {hanzi_model_path}")
        hanzi_checkpoint = torch.load(hanzi_model_path, map_location=self.device)
        self.hanzi_char_to_idx = hanzi_checkpoint['char_to_idx']
        self.hanzi_idx_to_char = hanzi_checkpoint['idx_to_char']
        self.hanzi_num_classes = hanzi_checkpoint['num_classes']
        self.hanzi_model = CNNModel(self.hanzi_num_classes).to(self.device)
        self.hanzi_model.load_state_dict(hanzi_checkpoint['model_state_dict'])
        self.hanzi_model.eval()
        
        if not os.path.exists(alnum_model_path):
            raise FileNotFoundError(f"字母数字模型文件不存在: {alnum_model_path}")
        alnum_checkpoint = torch.load(alnum_model_path, map_location=self.device)
        self.alnum_char_to_idx = alnum_checkpoint['char_to_idx']
        self.alnum_idx_to_char = alnum_checkpoint['idx_to_char']
        self.alnum_num_classes = alnum_checkpoint['num_classes']
        self.alnum_model = CNNModel(self.alnum_num_classes).to(self.device)
        self.alnum_model.load_state_dict(alnum_checkpoint['model_state_dict'])
        self.alnum_model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def save_debug_image(self, image, filename_prefix, description):
        timestamp = datetime.now().strftime("%H%M%S")
        filename = f"{filename_prefix}_{description}_{timestamp}.jpg"
        filepath = os.path.join(self.debug_output_dir, filename)
        if isinstance(image, Image.Image):
            image.save(filepath)
        elif isinstance(image, np.ndarray):
            cv2.imwrite(filepath, image)
        return filepath
    
    def gentle_smooth(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        smoothed = cv2.GaussianBlur(image, (3, 3), 0.5)
        return smoothed
    
    def simple_border_removal(self, image, debug_prefix=None):
        if isinstance(image, str):
            cv_image = cv2.imread(image)
        elif isinstance(image, Image.Image):
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):
            cv_image = image.copy()
        else:
            return image
        original = cv_image.copy()
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        mask = np.ones_like(gray) * 255
        edge_height = int(height * 0.05)
        top_edge = gray[0:edge_height, :]
        bottom_edge = gray[height-edge_height:height, :]
        top_black_ratio = np.sum(top_edge < 50) / (edge_height * width)
        bottom_black_ratio = np.sum(bottom_edge < 50) / (edge_height * width)
        if top_black_ratio > 0.6:
            mask[0:edge_height, :] = 0
        if bottom_black_ratio > 0.6:
            mask[height-edge_height:height, :] = 0
        result = cv2.bitwise_and(cv_image, cv_image, mask=mask)
        return result
    
    def detect_plate_color(self, image):
        if isinstance(image, str):
            cv_image = cv2.imread(image)
        elif isinstance(image, Image.Image):
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):
            cv_image = image.copy()
        else:
            return 'unknown'
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_ratio = np.sum(green_mask > 0) / (cv_image.shape[0] * cv_image.shape[1])
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([140, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        blue_ratio = np.sum(blue_mask > 0) / (cv_image.shape[0] * cv_image.shape[1])
        lower_yellow = np.array([15, 50, 50])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        yellow_ratio = np.sum(yellow_mask > 0) / (cv_image.shape[0] * cv_image.shape[1])
        color_ratios = {
            'green': green_ratio,
            'blue': blue_ratio,
            'yellow': yellow_ratio
        }
        max_color = max(color_ratios, key=color_ratios.get)
        max_ratio = color_ratios[max_color]
        if max_ratio < 0.1:
            return 'unknown'
        return max_color
    
    def clean_preprocess(self, image, debug_prefix=None):
        if isinstance(image, str):
            cv_image = cv2.imread(image)
        elif isinstance(image, Image.Image):
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):
            cv_image = image.copy()
        else:
            raise ValueError("不支持的图像格式")
        cv_image = self.simple_border_removal(cv_image, debug_prefix)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        smoothed = self.gentle_smooth(enhanced)
        _, binary = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        result = Image.fromarray(binary)
        return result
    
    def preprocess_for_color(self, image, color, debug_prefix=None):
        if color == 'green':
            return self.clean_preprocess(image, debug_prefix)
        return self.clean_preprocess(image, debug_prefix)
    
    def ensure_min_size(self, image, min_size=(48, 48)):
        if isinstance(image, str):
            pil_image = Image.open(image)
        elif isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            return image
        width, height = pil_image.size
        if width < min_size[0] or height < min_size[1]:
            ratio = max(min_size[0] / width, min_size[1] / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
        return pil_image

    def preprocess_image_with_retry(self, image, color, method='standard', debug_prefix=None):
        image = self.ensure_min_size(image, min_size=(48, 48))
        if method == 'standard':
            return self.preprocess_for_color(image, color, debug_prefix)
        elif method == 'inverted':
            processed = self.preprocess_for_color(image, color, debug_prefix)
            processed_np = np.array(processed)
            inverted = 255 - processed_np
            return Image.fromarray(inverted)
        elif method == 'high_contrast':
            cv_image = cv2.imread(image) if isinstance(image, str) else cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return Image.fromarray(binary)
        return self.preprocess_for_color(image, color, debug_prefix)
    
    def predict_with_model(self, image, model, idx_to_char, char_type='alnum', position=0):
        color = self.detect_plate_color(image)
        best_char = None
        best_confidence = 0
        best_method = 'standard'
        preprocess_methods = ['standard', 'inverted', 'high_contrast']
        for attempt in range(self.max_retries):
            if attempt < len(preprocess_methods):
                method = preprocess_methods[attempt]
            else:
                method = preprocess_methods[attempt % len(preprocess_methods)]
            try:
                debug_prefix = f"pos{position}_{char_type}_{method}_{attempt}"
                processed_image = self.preprocess_image_with_retry(image, color, method, debug_prefix)
                final_filename = f"pos{position}_{char_type}_final.jpg"
                final_filepath = os.path.join(self.debug_output_dir, final_filename)
                processed_image.save(final_filepath)
                input_tensor = self.transform(processed_image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)
                    confidence, predicted_idx = torch.max(probabilities, 0)
                char = idx_to_char[predicted_idx.item()]
                confidence_value = confidence.item()
                if confidence_value > best_confidence:
                    best_char = char
                    best_confidence = confidence_value
                    best_method = method
                if confidence_value >= self.confidence_threshold:
                    return char, confidence_value, color, method
            except Exception as e:
                continue
        if best_confidence > 0:
            return best_char, best_confidence, color, best_method
        else:
            return '?', 0.0, color, 'failed'
    
    def predict_hanzi(self, image, position=0):
        char, confidence, color, method = self.predict_with_model(
            image, self.hanzi_model, self.hanzi_idx_to_char, 'hanzi', position
        )
        return char, confidence, color, method
    
    def predict_alnum(self, image, position=0):
        char, confidence, color, method = self.predict_with_model(
            image, self.alnum_model, self.alnum_idx_to_char, 'alnum', position
        )
        return char, confidence, color, method
    
    def find_image_files(self, folder_path):
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif', '*.gif', '*.webp']
        image_files = []
        for extension in image_extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, extension)))
            image_files.extend(glob.glob(os.path.join(folder_path, extension.upper())))
        return image_files
    
    def extract_number_from_filename(self, filename):
        basename = os.path.basename(filename)
        name_without_ext = os.path.splitext(basename)[0]
        try:
            return int(name_without_ext)
        except ValueError:
            import re
            numbers = re.findall(r'\d+', name_without_ext)
            if numbers:
                return int(numbers[0])
            else:
                return 9999
    
    def recognize_plate_from_folder(self, folder_path='plate'):
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"文件夹不存在: {folder_path}")
        image_files = self.find_image_files(folder_path)
        if not image_files:
            raise FileNotFoundError(f"在 {folder_path} 中没有找到图片文件")
        file_mapping = {}
        for img_path in image_files:
            num = self.extract_number_from_filename(img_path)
            file_mapping[num] = img_path
        sorted_numbers = sorted(file_mapping.keys())
        first_image_path = file_mapping[sorted_numbers[0]]
        overall_color = self.detect_plate_color(first_image_path)
        results = []
        low_confidence_chars = 0
        for i, num in enumerate(sorted_numbers):
            img_path = file_mapping[num]
            if i == 0:
                char, confidence, color, method = self.predict_hanzi(img_path, position=i+1)
            else:
                char, confidence, color, method = self.predict_alnum(img_path, position=i+1)
            if confidence < self.confidence_threshold:
                low_confidence_chars += 1
            results.append({
                'position': i+1,
                'file': os.path.basename(img_path),
                'char': char,
                'confidence': confidence,
                'color': color,
                'method': method,
            })
        license_plate = ''.join([result['char'] for result in results])
        if overall_color == 'green' and len(results) == 8:
            formatted_plate = f"{license_plate[0]} {license_plate[1]} {license_plate[2:4]} {license_plate[4:6]} {license_plate[6:8]}"
        elif overall_color == 'blue' and len(results) == 7:
            formatted_plate = f"{license_plate[0]} {license_plate[1]} {license_plate[2:5]} {license_plate[5:7]}"
        else:
            formatted_plate = license_plate
        print(f"完整车牌号码: {license_plate}")
        return license_plate, results, overall_color


def main():

    hanzi_model = resource_path('models/hanzi_model.pth')
    alnum_model = resource_path('models/alnum_model.pth')
    

    if not os.path.exists(hanzi_model):
        print(f"汉字模型文件不存在: {hanzi_model}")
        return
    if not os.path.exists(alnum_model):
        print(f"字母数字模型文件不存在: {alnum_model}")
        return
        
    confidence_threshold = 0.9
    max_retries = 10
    try:
        recognizer = PlateRecognizer(
            hanzi_model, 
            alnum_model, 
            confidence_threshold=confidence_threshold,
            max_retries=max_retries,
            debug_output_dir='preprocess_debug'
        )
        license_plate, results, color = recognizer.recognize_plate_from_folder('plate')
        print(f"识别结果: {license_plate}")
    except Exception as e:
        print(f"识别失败: {e}")

if __name__ == "__main__":
    main()