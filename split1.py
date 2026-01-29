import cv2
import os
import numpy as np


def split_license_plate_v4(image_path, output_root_folder):
    # ================= 1. 路径与读取 =================
    if not os.path.exists(image_path):
        print(f"错误：找不到图片 {image_path}")
        return []

    file_name = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.join(output_root_folder, f"result_{file_name}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img = cv2.imread(image_path)
    if img is None:
        print("错误：图片读取失败")
        return []

    # ================= 2. 预处理优化 =================
    height, width = img.shape[:2]

    # 绿牌检测
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, np.array([35, 43, 46]), np.array([77, 255, 255]))
    is_green_plate = cv2.countNonZero(mask_green) > width * height * 0.2

    if is_green_plate:
        target_count = 8
    else:
        target_count = 7

    # 多种二值化方法尝试
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 方法1: 自适应二值化
    binary1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    # 方法2: Otsu二值化
    ret2, binary2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 方法3: 直接阈值（适用于蓝牌）
    ret3, binary3 = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    
    # 方法4: 反色二值化（适用于绿牌）
    ret4, binary4 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 根据车牌颜色选择二值化方法
    if is_green_plate:
        binary = binary4  # 绿牌用反色
    else:
        binary = binary2  # 蓝牌用Otsu

    # ================= 3. 形态学处理 =================
    # 添加边框
    border_size = 10
    binary_border = cv2.copyMakeBorder(binary, border_size, border_size, border_size, border_size, 
                                      cv2.BORDER_CONSTANT, value=0)
    img_border = cv2.copyMakeBorder(img, border_size, border_size, border_size, border_size,
                                   cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # 多种形态学处理尝试
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary_clean = cv2.morphologyEx(binary_border, cv2.MORPH_OPEN, kernel_clean)

    # 垂直方向膨胀连接字符
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    binary_processed = cv2.dilate(binary_clean, kernel_vertical, iterations=2)

    # ================= 4. 轮廓检测与筛选 =================
    def find_characters(binary_img, method_name):
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        char_candidates = []
        h_b, w_b = binary_img.shape[:2]

        # 合理的字符尺寸范围
        min_h = h_b * 0.3
        max_h = h_b * 0.9
        min_w = h_b * 0.08
        max_w = h_b * 0.4

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = w / h
            area = w * h
            
            # 基础筛选条件
            if (h > min_h and h < max_h and
                w > min_w and w < max_w and
                ratio > 0.1 and ratio < 1.2 and
                area > 80):
                char_candidates.append((x, y, w, h, area, ratio))

        # 按X坐标排序
        char_candidates.sort(key=lambda x: x[0])
        return char_candidates

    # 尝试不同的二值化图像
    candidates_list = []
    
    # 方法1: 处理后的二值图像
    candidates1 = find_characters(binary_processed, "处理后二值图")
    if candidates1:
        candidates_list.append(("processed", candidates1))
    
    # 方法2: 原始二值图像
    binary_border_orig = cv2.copyMakeBorder(binary, border_size, border_size, border_size, border_size, 
                                           cv2.BORDER_CONSTANT, value=0)
    candidates2 = find_characters(binary_border_orig, "原始二值图")
    if candidates2:
        candidates_list.append(("original", candidates2))
    
    # 方法3: 自适应二值化
    binary_adaptive_border = cv2.copyMakeBorder(binary1, border_size, border_size, border_size, border_size, 
                                               cv2.BORDER_CONSTANT, value=0)
    candidates3 = find_characters(binary_adaptive_border, "自适应二值图")
    if candidates3:
        candidates_list.append(("adaptive", candidates3))

    # 选择最好的候选结果
    if not candidates_list:
        # 最后尝试：直接阈值
        _, binary_last = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        binary_last_border = cv2.copyMakeBorder(binary_last, border_size, border_size, border_size, border_size, 
                                               cv2.BORDER_CONSTANT, value=0)
        candidates_last = find_characters(binary_last_border, "最后尝试")
        if candidates_last:
            candidates_list.append(("last", candidates_last))

    if not candidates_list:
        return []

    # 选择字符数量最接近目标的方法
    best_candidates = None
    best_method = ""
    min_diff = float('inf')
    
    for method_name, candidates in candidates_list:
        diff = abs(len(candidates) - target_count)
        if diff < min_diff:
            min_diff = diff
            best_candidates = candidates
            best_method = method_name

    # ================= 5. 字符筛选和优化 =================
    if best_candidates:
        # 提取坐标信息
        char_candidates = [(x, y, w, h) for x, y, w, h, area, ratio in best_candidates]
        
        # 基于统计信息的筛选
        heights = [h for _, _, _, h in char_candidates]
        widths = [w for _, _, w, _ in char_candidates]
        ys = [y for _, y, _, _ in char_candidates]
        
        median_h = np.median(heights)
        median_y = np.median(ys)
        median_w = np.median(widths)

        # 筛选条件
        filtered_chars = []
        for x, y, w, h in char_candidates:
            height_ratio = abs(h - median_h) / median_h
            y_diff = abs(y - median_y)
            
            if height_ratio < 0.4 and y_diff < 30:
                filtered_chars.append((x, y, w, h))

        char_candidates = filtered_chars

        # 如果字符数量仍然不匹配，进行调整
        if len(char_candidates) > target_count:
            # 按面积排序，保留最大的target_count个
            char_candidates.sort(key=lambda c: c[2] * c[3], reverse=True)
            char_candidates = char_candidates[:target_count]
            char_candidates.sort(key=lambda x: x[0])
        elif len(char_candidates) < target_count:
            pass  # 字符不足，不做处理

    else:
        char_candidates = []
        return []

    # ================= 6. 保存结果 =================
    for i, (x, y, w, h) in enumerate(char_candidates):
        pad = 3
        roi = img_border[max(0, y-pad):min(img_border.shape[0], y+h+pad), 
                         max(0, x-pad):min(img_border.shape[1], x+w+pad)]
        save_path = os.path.join(output_dir, f"{i + 1}.png")
        cv2.imwrite(save_path, roi)

    return char_candidates


if __name__ == "__main__":
    input_img = "output_results/1.jpg"  # 图片路径
    output_root = "picture_output"

    split_license_plate_v4(input_img, output_root)