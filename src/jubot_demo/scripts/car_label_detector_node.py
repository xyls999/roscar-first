#!/usr/bin/env python3
# coding=utf-8
# 车牌检测节点
# 功能：使用PaddleOCR识别车牌号码，发布识别结果到ROS话题

import rospy
from std_msgs.msg import Bool, String
import os
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
from paddleocr import PaddleOCR
import re

class CarLabelDetectorNode:
    """车牌检测节点类"""
    def __init__(self):
        # 初始化车牌检测节点
        rospy.init_node('car_label_detector_node', anonymous=True)
        
        # 从参数服务器获取配置参数，设置模型和图像路径
        self.base_dir = rospy.get_param('~base_dir', os.path.join(os.path.expanduser('~'), 'ros1', 'images', 'car'))
        self.model_dir = rospy.get_param('~model_dir', os.path.join(os.path.expanduser('~'), 'ros1', 'model_car_label'))
        self.input_image = rospy.get_param('~input_image', os.path.join(self.base_dir, "car_1.jpg"))
        
        # 订阅车牌检测触发话题，接收检测指令
        rospy.Subscriber("/detect/car_label", Bool, self.car_label_callback)
        
        # 创建发布器，将识别结果发布到/words话题供其他节点使用
        self.words_pub = rospy.Publisher('/words', String, queue_size=10)
        
        rospy.loginfo("Car label detector node started")
    
    def car_label_callback(self, msg):
        """接收到/detect/car_label消息时的回调函数"""
        # 功能：响应检测触发信号，启动车牌识别流程
        if msg.data:  # 只有当消息为True时才执行
            rospy.loginfo("接收到车牌检测请求，开始执行车牌识别...")
            self.run_license_plate_recognition()
    
    def run_license_plate_recognition(self):
        """执行车牌识别的主程序"""
        # 功能：完整的车牌识别流程，包括图像检查和结果发布
        # 检查输入图片是否存在
        if not os.path.exists(self.input_image):
            rospy.logerr(f"错误: 输入图片不存在: {self.input_image}")
            return
        
        # 车牌识别
        plate_number = self.recognize_plate(self.input_image)
        
        # 发布识别结果
        if plate_number:
            message = f"车牌号为{plate_number}"
            words_msg = String()
            words_msg.data = message
            self.words_pub.publish(words_msg)
            rospy.loginfo(f"发布识别结果: {message}")
        else:
            rospy.loginfo("未识别到车牌号")
    
    def recognize_plate(self, input_path):
        """车牌识别函数，使用指定目录的模型"""
        # 功能：使用PaddleOCR进行车牌文字识别，返回识别结果
        try:
            # 定义PaddleOCR模型目录结构
            model_dirs = {
                'det': os.path.join(self.model_dir, 'det', 'ch', 'ch_PP-OCRv3_det_infer'),
                'rec': os.path.join(self.model_dir, 'rec', 'ch', 'ch_PP-OCRv3_rec_infer'),
                'cls': os.path.join(self.model_dir, 'cls', 'ch_ppocr_mobile_v2.0_cls_infer')
            }
            
            # 检查模型是否存在
            for model_type, model_path in model_dirs.items():
                model_file = os.path.join(model_path, 'inference.pdmodel')
                if not os.path.exists(model_file):
                    rospy.logerr(f"错误: {model_type} 模型不存在于 {model_path}")
                    return None
            
            # 初始化PaddleOCR，使用指定路径的模型
            ocr = PaddleOCR(
                use_angle_cls=True, 
                lang='ch',
                use_gpu=False,
                det_model_dir=model_dirs['det'],
                rec_model_dir=model_dirs['rec'], 
                cls_model_dir=model_dirs['cls'],
                show_log=False  # 关闭详细日志，减少输出
            )
            
            # 识别图片
            result = ocr.ocr(input_path, cls=True)
            
            if result and result[0]:
                # 提取所有文本
                texts = [line[1][0] for line in result[0] if line]
                
                # 查找最像车牌号的文本
                for text in texts:
                    # 清理文本并转为大写
                    clean_text = re.sub(r'[^A-Z0-9京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼]', '', text.upper())
                    
                    # 简单判断是否为车牌（长度5-7位，包含字母和数字）
                    if 5 <= len(clean_text) <= 7 and any(c.isalpha() for c in clean_text) and any(c.isdigit() for c in clean_text):
                        return clean_text
                
                # 如果没有找到符合的，返回第一个结果
                if texts:
                    return re.sub(r'[^A-Z0-9京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼]', '', texts[0].upper())
            
            return None
            
        except Exception as e:
            rospy.logerr(f"车牌识别错误: {e}")
            return None

def main():
    try:
        node = CarLabelDetectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"节点运行错误: {e}")

if __name__ == '__main__':
    main()