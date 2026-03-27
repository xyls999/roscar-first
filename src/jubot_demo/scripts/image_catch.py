#!/usr/bin/env python3
# coding=utf-8
# 图像捕获节点
# 功能：从摄像头捕获图像并触发检测任务，支持图像增强和自动检测指令发布
import rospy
import os
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Empty, Bool
import threading

class CameraCapture:
    """摄像头捕获类"""
    def __init__(self):
        self.photo_count = 0
        # 初始化摄像头捕获节点
        rospy.init_node('camera_capture_node', anonymous=True)
        
        # 设置图像保存的基础路径
        self.base_path = os.path.join(os.path.expanduser('~'), 'ros1', 'images')
        
        # 创建图像存储目录，分别存储人员检测和车辆检测图像
        self.person_dir = os.path.join(self.base_path, "person")
        self.car_dir = os.path.join(self.base_path, "car")
        
        os.makedirs(self.person_dir, exist_ok=True)
        os.makedirs(self.car_dir, exist_ok=True)
        
        # 订阅摄像头图像话题和拍照触发话题
        self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.image_callback)
        self.trigger_sub = rospy.Subscriber("/robot/take_photo", Empty, self.capture_and_save_image_callback)
        
        # 创建检测指令发布器，用于触发不同类型的检测任务
        self.detect_person_one_pub = rospy.Publisher('/detect/person/one', Bool, queue_size=1)
        self.detect_person_two_pub = rospy.Publisher('/detect/person/two', Bool, queue_size=1)
        self.detect_car_label_pub = rospy.Publisher('/detect/car_label', Bool, queue_size=1)
        
        # 线程安全机制，保护图像数据访问
        self.latest_image_msg = None
        self.image_lock = threading.Lock()
        
        # 拍照计数器，用于确定保存路径和触发相应的检测任务
        self.photo_count = 0
        
        rospy.loginfo("拍照节点已启动")

    def image_callback(self, data):
        # 功能：接收并缓存最新的摄像头图像数据
        with self.image_lock:
            self.latest_image_msg = data
    
    def image_msg_to_cv2(self, image_msg):
        """
        将ROS Image消息转换为OpenCV图像
        功能：处理不同编码格式的ROS图像消息，转换为OpenCV可处理的格式
        """
        try:
            # 检查图像编码格式
            if image_msg.encoding == "bgr8" or image_msg.encoding == "rgb8":
                # 直接将数据转换为numpy数组
                if image_msg.encoding == "rgb8":
                    # 如果是RGB，需要转换为BGR
                    cv_image = np.frombuffer(image_msg.data, dtype=np.uint8).reshape(
                        image_msg.height, image_msg.width, 3)
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
                else:
                    # BGR格式直接使用
                    cv_image = np.frombuffer(image_msg.data, dtype=np.uint8).reshape(
                        image_msg.height, image_msg.width, 3)
                return cv_image
            elif image_msg.encoding == "mono8":
                # 单通道灰度图像
                cv_image = np.frombuffer(image_msg.data, dtype=np.uint8).reshape(
                    image_msg.height, image_msg.width)
                return cv_image
            else:
                rospy.logwarn(f"不支持的图像编码格式: {image_msg.encoding}")
                return None
        except Exception as e:
            rospy.logerr(f"转换图像格式时出错: {e}")
            return None
            
    def enhance_image(self, image):
        """
        对图像进行增强，主要增加饱和度。
        功能：提升图像质量，增强检测效果，包括饱和度调整和对比度增强
        """
        try:
            # 检查图像是否为单通道
            if len(image.shape) == 2:
                # 如果是灰度图，转换为BGR
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            # 将图像从BGR色彩空间转换到HSV色彩空间
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # 分离H, S, V三个通道
            h, s, v = cv2.split(hsv_image)
            
            # 增加饱和度
            saturation_increase = 50
            lut = np.array([i + saturation_increase if i + saturation_increase < 255 else 255 for i in np.arange(0, 256)]).astype("uint8")
            
            # 应用查找表到饱和度通道
            s_enhanced = cv2.LUT(s, lut)
            
            # 轻微增加对比度
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            v_enhanced = clahe.apply(v)
            
            # 合并增强后的通道
            hsv_enhanced = cv2.merge([h, s_enhanced, v_enhanced])
            
            # 将图像从HSV转换回BGR色彩空间
            enhanced_image = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
            
            return enhanced_image
        except Exception as e:
            rospy.logwarn(f"图像增强失败，返回原图")
            return image

    def capture_and_save_image_callback(self, msg):
        # 功能：响应拍照触发信号，捕获当前图像并保存，同时发布相应的检测指令
        with self.image_lock:
            if self.latest_image_msg is None:
                rospy.logwarn("尚未接收到图像数据，无法拍照！")
                return
            image_data = self.latest_image_msg
        
        try:
            # 转换图像
            original_image = self.image_msg_to_cv2(image_data)
            
            if original_image is None:
                rospy.logerr("无法转换图像数据")
                return
            
            # 对图像进行增强
            enhanced_image = self.enhance_image(original_image)
            
            # 根据拍照次数确定保存路径和文件名，实现自动分类存储
            self.photo_count += 1
            
            if self.photo_count == 1:
                # 第一张照片：人员检测图像1
                filename = "person_1.jpg"
                filepath = os.path.join(self.person_dir, filename)
            elif self.photo_count == 2:
                # 第二张照片：人员检测图像2
                filename = "person_2.jpg"
                filepath = os.path.join(self.person_dir, filename)
            elif self.photo_count == 3:
                # 第三张照片：车辆检测图像
                filename = "car_1.jpg"
                filepath = os.path.join(self.car_dir, filename)
            else:
                # 超过三次，使用默认命名存储
                filename = f"extra_{self.photo_count}.jpg"
                filepath = os.path.join(self.base_path, filename)
            
            # 保存增强后的图片
            success = cv2.imwrite(filepath, enhanced_image)
            
            if success:
                rospy.loginfo(f"图片保存成功: {filename}")
                
                # 根据拍照次数发布相应的检测指令，实现自动化检测流程
                if self.photo_count == 1:
                    rospy.loginfo("发布人员检测指令到 /detect/person/one")
                    msg = Bool()
                    msg.data = True
                    self.detect_person_one_pub.publish(msg)
                elif self.photo_count == 2:
                    rospy.loginfo("发布人员检测指令到 /detect/person/two")
                    msg = Bool()
                    msg.data = True
                    self.detect_person_two_pub.publish(msg)
                elif self.photo_count == 3:
                    rospy.loginfo("发布车辆检测指令到 /detect/car_label")
                    msg = Bool()
                    msg.data = True
                    self.detect_car_label_pub.publish(msg)
                else:
                    rospy.loginfo(f"第 {self.photo_count} 次拍照，不发布检测指令")
                    
            else:
                rospy.logerr(f"保存图片失败: {filename}")

        except Exception as e:
            rospy.logerr(f"处理图像时发生错误: {e}")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        capture_node = CameraCapture()
        capture_node.run()
    except rospy.ROSInterruptException:
        pass