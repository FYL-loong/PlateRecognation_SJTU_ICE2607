import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import customtkinter as ctk
import cv2
import numpy as np
import os
import glob
import sys
from threading import Thread
from plate_detector import detect_plate
from split1 import split_license_plate_v4
from recognize_plate import PlateRecognizer

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

def resource_path(relative_path):
    """获取资源的绝对路径，用于PyInstaller打包后访问资源文件"""
    try:
        # PyInstaller创建临时文件夹，将路径存储在_MEIPASS中
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class LicensePlateRecognitionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("车牌识别系统")
        self.root.geometry("1000x700")
        self.root.resizable(True, True)
        
        self.current_image = None
        self.current_image_path = None
        self.original_image = None
        
        self.setup_ui()
        
    def setup_ui(self):
        main_container = ctk.CTkFrame(self.root, fg_color="transparent")
        main_container.pack(fill="both", expand=True, padx=20, pady=20)
        
        title_label = ctk.CTkLabel(
            main_container,
            text="🚗 车牌识别系统",
            font=ctk.CTkFont(size=32, weight="bold")
        )
        title_label.pack(pady=(0, 30))
        
        content_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        content_frame.pack(fill="both", expand=True)
        
        left_frame = ctk.CTkFrame(content_frame, width=500)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        left_frame.pack_propagate(False)
        
        upload_frame = ctk.CTkFrame(left_frame)
        upload_frame.pack(pady=20, padx=20, fill="x")
        
        upload_btn = ctk.CTkButton(
            upload_frame,
            text="📁 选择图片",
            command=self.upload_image,
            height=50,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#1f538d",
            hover_color="#14375e"
        )
        upload_btn.pack(pady=10, padx=20, fill="x")
        
        preview_label = ctk.CTkLabel(
            left_frame,
            text="图片预览区域",
            font=ctk.CTkFont(size=14),
            fg_color="#2b2b2b",
            corner_radius=10
        )
        preview_label.pack(pady=20, padx=20, fill="both", expand=True)
        self.preview_label = preview_label
        
        right_frame = ctk.CTkFrame(content_frame, width=500)
        right_frame.pack(side="right", fill="both", expand=True, padx=(10, 0))
        right_frame.pack_propagate(False)
        
        recognize_btn = ctk.CTkButton(
            right_frame,
            text="🔍 开始识别",
            command=self.recognize_plate,
            height=50,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#2d8659",
            hover_color="#1f5c3f",
            state="disabled"
        )
        recognize_btn.pack(pady=20, padx=20, fill="x")
        self.recognize_btn = recognize_btn
        
        result_frame = ctk.CTkFrame(right_frame)
        result_frame.pack(pady=20, padx=20, fill="both", expand=True)
        
        result_title = ctk.CTkLabel(
            result_frame,
            text="识别结果",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        result_title.pack(pady=(20, 10))
        
        self.result_text = ctk.CTkTextbox(
            result_frame,
            height=400,
            font=ctk.CTkFont(size=18),
            wrap="word"
        )
        self.result_text.pack(pady=10, padx=20, fill="both", expand=True)
        
        self.status_label = ctk.CTkLabel(
            main_container,
            text="就绪",
            font=ctk.CTkFont(size=12),
            fg_color="#2b2b2b",
            corner_radius=5
        )
        self.status_label.pack(pady=(10, 0), fill="x")
    
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[
                ("图片文件", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("所有文件", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.current_image_path = file_path
                image = Image.open(file_path)
                self.original_image = cv2.imread(file_path)
                
                preview_width = 450
                preview_height = 400
                image.thumbnail((preview_width, preview_height), Image.Resampling.LANCZOS)
                
                photo = ImageTk.PhotoImage(image)
                self.preview_label.configure(image=photo, text="")
                self.preview_label.image = photo
                
                self.current_image = image
                self.recognize_btn.configure(state="normal")
                self.update_status(f"已加载图片: {os.path.basename(file_path)}")
                self.result_text.delete("1.0", "end")
                
            except Exception as e:
                messagebox.showerror("错误", f"加载图片失败: {str(e)}")
                self.update_status("图片加载失败")
    
    def recognize_plate(self):
        if not self.current_image_path:
            messagebox.showwarning("警告", "请先选择图片")
            return
        
        self.recognize_btn.configure(state="disabled", text="识别中...")
        self.update_status("正在识别车牌，请稍候...")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", "正在识别中，请稍候...")
        
        thread = Thread(target=self._recognize_plate_thread)
        thread.daemon = True
        thread.start()
    
    def _recognize_plate_thread(self):
        try:
            # 清理临时文件夹
            self._clean_temp_folders()
            
            if self.original_image is None:
                raise ValueError("图片未加载")
            
            # 创建临时文件夹
            os.makedirs('temp_output', exist_ok=True)
            os.makedirs('temp_plate', exist_ok=True)
            
            # 步骤1: 检测车牌区域
            self.root.after(0, self.update_status, "正在检测车牌区域...")
            plate_output_path = 'temp_output/plate.jpg'
            bbox = detect_plate(self.current_image_path, plate_output_path)
            
            if not os.path.exists(plate_output_path):
                self.root.after(0, self._show_no_plate)
                return
            
            # 步骤2: 分割字符
            self.root.after(0, self.update_status, "正在分割字符...")
            char_candidates = split_license_plate_v4(plate_output_path, 'temp_plate')
            
            # 查找分割后的字符文件夹
            split_folders = glob.glob('temp_plate/result_*')
            if not split_folders:
                self.root.after(0, self._show_no_plate)
                return
            
            split_folder = split_folders[0]
            
            # 检查字符图片数量
            char_images = glob.glob(os.path.join(split_folder, '*.png'))
            if len(char_images) < 5:  # 至少需要5个字符（汉字+字母数字）
                self.root.after(0, self._show_no_plate)
                return
            
            # 步骤3: 识别字符
            self.root.after(0, self.update_status, "正在识别字符...")

            # 创建字符识别器，使用resource_path获取模型路径
            recognizer = PlateRecognizer(
                hanzi_model_path=resource_path('models/hanzi_model.pth'),
                alnum_model_path=resource_path('models/alnum_model.pth'),
                confidence_threshold=0.9,
                max_retries=10,
                debug_output_dir='preprocess_debug'
            )

            # 直接调用字符识别方法，接收3个返回值（保持原样）
            license_plate, results, color = recognizer.recognize_plate_from_folder(split_folder)
            
            # 提取纯车牌号码（去掉"完整车牌号码: "前缀）
            if license_plate.startswith("完整车牌号码: "):
                license_plate = license_plate.replace("完整车牌号码: ", "")
            
            # 添加颜色名称映射
            color_names = {
                'green': '绿色',
                'blue': '蓝色',
                'yellow': '黄色',
                'unknown': '未知颜色'
            }
            
            color_name = color_names.get(color, '未知颜色')
            
            # 传递颜色信息给结果更新方法
            self.root.after(0, self._update_results, license_plate, color_name)
            
        except Exception as e:
            error_msg = f"识别失败: {str(e)}"
            self.root.after(0, self._show_error, error_msg)
    
    def _clean_temp_folders(self):
        """清理临时文件夹"""
        import shutil
        temp_folders = ['temp_output', 'temp_plate', 'preprocess_debug']
        for folder in temp_folders:
            if os.path.exists(folder):
                shutil.rmtree(folder)
    
    def _update_results(self, plate_text, color_name=""):
        self.recognize_btn.configure(state="normal", text="🔍 开始识别")
        
        if plate_text and len(plate_text.strip()) > 0 and plate_text != "?" * len(plate_text):
            # 添加颜色信息到结果显示
            if color_name:
                result_str = f"✅ 识别到的车牌号码：\n\n{plate_text}\n\n车牌类型：{color_name}车牌"
            else:
                result_str = f"✅ 识别到的车牌号码：\n\n{plate_text}"
            
            self.result_text.delete("1.0", "end")
            self.result_text.insert("1.0", result_str)
            self.update_status("识别完成！")
        else:
            self._show_no_plate()
    
    def _show_no_plate(self):
        self.recognize_btn.configure(state="normal", text="🔍 开始识别")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", "❌ 未检测到车牌区域\n\n请确保图片中包含清晰可见的车牌。")
        self.update_status("未检测到车牌")
    
    def _show_error(self, error_msg):
        self.recognize_btn.configure(state="normal", text="🔍 开始识别")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", f"错误: {error_msg}")
        self.update_status("识别失败")
        messagebox.showerror("错误", error_msg)
    
    def update_status(self, message):
        self.status_label.configure(text=f"状态: {message}")

def main():
    root = ctk.CTk()
    app = LicensePlateRecognitionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()