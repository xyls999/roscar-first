#!/usr/bin/env python3
# coding=utf-8
# 人员检测节点
# 功能：基于ONNX模型进行人员检测和分类，支持多检测器实例

import rospy
from std_msgs.msg import Bool, String
import os
import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort
import threading

class PersonDetectorNode:
    def __init__(self):
        # 初始化人员检测节点
        # 获取节点名称和检测器ID，支持多实例部署
        self.node_name = rospy.get_name()
        self.detector_id = rospy.get_param('~detector_id', '1')  # 检测器标识符
        
        # 根据检测器ID设置不同的默认参数
        # 功能：支持多个检测器实例，每个实例处理不同街道的人员检测
        if self.detector_id == '1':
            # 检测器1：处理A街的人员检测
            default_image = os.path.join(os.path.expanduser('~'), 'ros1', 'images', 'person', 'person_1.jpg')
            default_street = "A街"
            default_topic = "/detect/person/one"
        else:
            # 检测器2：处理B街的人员检测
            default_image = os.path.join(os.path.expanduser('~'), 'ros1', 'images', 'person', 'person_2.jpg')
            default_street = "B街"
            default_topic = "/detect/person/two"
        
        # 从ROS参数服务器获取配置参数，支持运行时参数调整
        self.input_image = rospy.get_param('~input_image', default_image)
        self.output_directory = rospy.get_param('~output_directory', 
                                              os.path.join(os.path.expanduser('~'), 'ros1', 'images', 'person'))
        self.model_path = rospy.get_param('~model_path', 
                                         os.path.join(os.path.expanduser('~'), 'ros1', 'person.onnx'))
        self.street_name = rospy.get_param('~street_name', default_street)
        self.subscribe_topic = rospy.get_param('~subscribe_topic', default_topic)
        
        # 订阅检测触发话题，接收检测指令
        rospy.Subscriber(self.subscribe_topic, Bool, self.person_callback)
        
        # 创建发布器，将检测结果发布到/words话题供其他节点使用
        self.words_pub = rospy.Publisher('/words', String, queue_size=10)
        
        # 线程同步机制，防止多个检测任务同时执行，确保线程安全
        self.processing_lock = threading.Lock()
        self.is_processing = False  # 标记当前是否正在处理图像
        
        rospy.loginfo(f"Person detector {self.detector_id} node started, subscribing to {self.subscribe_topic}")
        
    def person_callback(self, msg):
        # 接收到检测话题时的回调函数
        # 功能：响应检测触发信号，启动异步图像处理流程
        if msg.data:  # 只有当消息为True时才处理
            rospy.loginfo(f"检测器{self.detector_id}接收到检测请求，开始处理图像...")
            
            # 使用独立线程处理图像，避免阻塞ROS主循环
            # 功能：保持ROS节点的响应性，支持并发处理
            if not self.is_processing:
                thread = threading.Thread(target=self.process_image_thread)
                thread.daemon = True
                thread.start()
            else:
                rospy.logwarn(f"检测器{self.detector_id}当前正在处理图像，忽略新的请求")
    
    def process_image_thread(self):
        # 图像处理线程函数
        # 获取处理锁，确保线程安全
        with self.processing_lock:
            if self.is_processing:
                return  # 如果已经在处理，直接返回
            self.is_processing = True
        
        try:
            # 执行完整的图像处理管道
            success = self.process_image_pipeline()
            if success:
                rospy.loginfo(f"检测器{self.detector_id}图像处理完成!")
            else:
                rospy.logerr(f"检测器{self.detector_id}图像处理失败!")
        except Exception as e:
            rospy.logerr(f"检测器{self.detector_id}图像处理过程中出现错误: {e}")
        finally:
            # 无论成功还是失败，都要释放处理锁
            self.is_processing = False
    
    def process_image_pipeline(self):
        # 主要图像处理管道函数
        # 确保输出目录存在，如果不存在则创建
        os.makedirs(self.output_directory, exist_ok=True)
        
        # 生成带时间戳的中间文件路径，避免文件名冲突
        base_name = os.path.splitext(os.path.basename(self.input_image))[0]
        timestamp = rospy.Time.now().to_nsec()  # 使用纳秒级时间戳确保唯一性
        
        # 定义中间处理文件的路径
        change_path = os.path.join(self.output_directory, f"{base_name}_change_{timestamp}.jpg")
        detect_path = os.path.join(self.output_directory, f"{base_name}_detected_{timestamp}.jpg")
        
        # 第一步：图像尺寸调整和标准化
        try:
            rospy.loginfo(f"开始调整图像尺寸: {self.input_image}")
            self.change_image(self.input_image, change_path)
            rospy.loginfo(f"图像尺寸调整完成: {change_path}")
        except Exception as e:
            rospy.logerror(f"检测器{self.detector_id}尺寸调整失败: {e}")
            return False
        
        # 第二步：目标检测和分类
        try:
            rospy.loginfo(f"开始目标检测: {change_path}")
            detections, class_counts = self.detect_objects(change_path, detect_path, self.model_path)
            rospy.loginfo(f"目标检测完成，检测到 {len(detections)} 个目标")
            
            # 发布检测结果到ROS话题
            self.publish_detection_result(class_counts)
            
        except Exception as e:
            rospy.logerror(f"检测器{self.detector_id}目标检测失败: {e}")
            return False
        
        return True
    
    def publish_detection_result(self, class_counts):
        # Publish detection results to /words topic
        citizen_count = class_counts.get(0, 0)
        not_citizen_count = class_counts.get(1, 0)
        
        # 格式化消息
        message = f"{self.street_name}有公民{citizen_count}人，非公民{not_citizen_count}人"
        
        # 发布到/words话题
        words_msg = String()
        words_msg.data = message
        self.words_pub.publish(words_msg)
        
        rospy.loginfo(f"检测器{self.detector_id}发布检测结果: {message}")
    
    # 以下是图像处理函数的定义
    
    def change_image(self, input_path, output_path):
        # 图像尺寸调整函数
        original_image = Image.open(input_path)
        original_width, original_height = original_image.size
        scale = min(640 / original_width, 640 / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        resized_image = original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        new_image = Image.new('RGB', (640, 640))
        paste_x = (640 - new_width) // 2
        paste_y = (640 - new_height) // 2
        fill_color = self.get_smart_fill_color(resized_image)
        new_image.paste(fill_color, [0, 0, 640, 640])
        new_image.paste(resized_image, (paste_x, paste_y))
        new_image.save(output_path, quality=95)
    
    def get_smart_fill_color(self, image):
        # 智能填充颜色选择
        img_array = np.array(image)
        edge_pixels = []
        height, width = img_array.shape[:2]
        
        if width > 10 and height > 10:
            edge_pixels.extend(img_array[0:5, :].reshape(-1, 3))
            edge_pixels.extend(img_array[-5:, :].reshape(-1, 3))
            edge_pixels.extend(img_array[:, 0:5].reshape(-1, 3))
            edge_pixels.extend(img_array[:, -5:].reshape(-1, 3))
        
        if len(edge_pixels) > 0:
            edge_pixels = np.array(edge_pixels)
            edge_color = tuple(np.median(edge_pixels, axis=0).astype(int))
            brightness = np.mean(edge_color)
            if brightness < 30:
                return (50, 50, 50)
            elif brightness > 220:
                return (200, 200, 200)
            else:
                return edge_color
        else:
            avg_color = tuple(np.median(img_array.reshape(-1, 3), axis=0).astype(int))
            brightness = np.mean(avg_color)
            if brightness < 50:
                return (80, 80, 80)
            elif brightness > 200:
                return (180, 180, 180)
            else:
                return tuple((np.array(avg_color) * 0.7 + 255 * 0.3).astype(int))
    
    def detect_objects(self, input_path, out_path, model_path):
        # 使用ONNX模型的主要目标检测函数
        # 加载ONNX模型
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        
        # 读取输入图像
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"无法读取图片: {input_path}")
        
        # 确保图像尺寸为640x640
        h, w = image.shape[:2]
        if h != 640 or w != 640:
            image = cv2.resize(image, (640, 640))
            h, w = 640, 640
        
        def preprocess(image):
            # 图像预处理函数
            # 归一化到0-1范围
            image = image.astype(np.float32) / 255.0
            # BGR转RGB
            image = image[:, :, ::-1]
            # HWC转CHW格式
            image = image.transpose(2, 0, 1)
            # 添加batch维度
            image = np.expand_dims(image, axis=0)
            return image
        
        # 预处理图像并运行模型推理
        input_tensor = preprocess(image)
        outputs = session.run(None, {input_name: input_tensor})
        
        # 后处理检测结果
        detections = self.postprocess_detections(outputs)
        
        # 统计各类别检测数量
        class_counts = {0: 0, 1: 0}  # 0:公民, 1:非公民
        for detection in detections:
            class_id = detection['class']
            if class_id in class_counts:
                class_counts[class_id] += 1
        
        # 绘制检测结果并保存
        result_image = self.draw_detections(image, detections, class_counts)
        cv2.imwrite(out_path, result_image)
        
        return detections, class_counts
    
    def postprocess_detections(self, outputs, conf_threshold=0.25, iou_threshold=0.45):
        # 使用NMS后处理检测结果
        predictions = outputs[0]
        # 处理batch维度
        if len(predictions.shape) == 3:
            predictions = predictions[0]
        
        detections = []
        
        # 遍历所有预测结果
        for pred in predictions:
            if len(pred) < 6:
                continue
                
            # 解析边界框坐标和置信度
            x, y, w, h = pred[0:4]  # 中心点坐标和宽高
            conf = pred[4]  # 目标置信度
            
            # 过滤低置信度检测
            if conf < conf_threshold:
                continue
            
            # 解析类别概率
            class_probs = pred[5:]
            class_id = np.argmax(class_probs)
            class_score = class_probs[class_id]
            final_conf = conf * class_score  # 最终置信度
            
            # 转换为中心点格式到角点格式
            x1 = int(x - w/2)
            y1 = int(y - h/2)
            x2 = int(x + w/2)
            y2 = int(y + h/2)
            
            # 边界检查，确保坐标在图像范围内
            x1 = max(0, min(x1, 639))
            y1 = max(0, min(y1, 639))
            x2 = max(0, min(x2, 639))
            y2 = max(0, min(y2, 639))
            
            # 验证边界框有效性
            if x2 > x1 and y2 > y1 and final_conf >= conf_threshold:
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(final_conf),
                    'class': int(class_id)
                })
        
        # 应用非极大值抑制去除重复检测
        return self.non_max_suppression(detections, iou_threshold)
    
    def non_max_suppression(self, detections, iou_threshold):
        # 非极大值抑制算法
        if len(detections) == 0:
            return []
        
        # 按置信度降序排序
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while detections:
            # 选择置信度最高的检测框
            current = detections.pop(0)
            keep.append(current)
            
            # 找到与当前检测框IoU超过阈值的检测框
            to_remove = []
            for i, det in enumerate(detections):
                iou = self.calculate_iou(current['bbox'], det['bbox'])
                if iou > iou_threshold:
                    to_remove.append(i)
            
            # 从后往前删除，避免索引变化
            for i in sorted(to_remove, reverse=True):
                detections.pop(i)
        
        return keep
    
    def calculate_iou(self, box1, box2):
        # 计算两个边界框的IoU
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # 计算交集区域的坐标
        xi1 = max(x1_1, x1_2)  # 交集左边界
        yi1 = max(y1_1, y1_2)  # 交集上边界
        xi2 = min(x2_1, x2_2)  # 交集右边界
        yi2 = min(y2_1, y2_2)  # 交集下边界
        
        # 计算交集面积
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # 计算两个边界框的面积
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # 计算并集面积
        union_area = box1_area + box2_area - inter_area
        
        # 返回IoU值，避免除零错误
        return inter_area / union_area if union_area > 0 else 0
    
    def draw_detections(self, image, detections, class_counts=None):
        # 在图像上绘制检测结果
        result_image = image.copy()
        
        # 定义不同类别的颜色（BGR格式）
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
                  (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)]
        
        # 类别名称映射
        class_names = {0: "Citizen", 1: "not_Citizen"}
        
        # 在图像左上角添加类别统计信息
        if class_counts is not None:
            y_offset = 30
            for class_id, count in class_counts.items():
                class_name = class_names.get(class_id, f"Class {class_id}")
                text = f"{class_name}: {count}"
                cv2.putText(result_image, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_offset += 30
        
        # 绘制每个检测结果
        for det in detections:
            bbox = det['bbox']
            conf = det['confidence']
            cls = det['class']
            color = colors[cls % len(colors)]  # 根据类别选择颜色
            
            # 绘制边界框
            cv2.rectangle(result_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # 准备标签文本
            class_name = class_names.get(cls, f"Class {cls}")
            label = f"{class_name}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # 绘制标签背景
            cv2.rectangle(result_image, 
                         (bbox[0], bbox[1] - label_size[1] - 10),
                         (bbox[0] + label_size[0], bbox[1]),
                         color, -1)
            
            # 绘制标签文本
            cv2.putText(result_image, label, 
                       (bbox[0], bbox[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return result_image

def main():
    # 主函数：初始化ROS节点并启动人员检测器
    try:
        # 初始化ROS节点
        rospy.init_node('person_detector_node', anonymous=True)
        
        # 创建人员检测节点实例
        node = PersonDetectorNode()
        
        # 保持节点运行，等待消息
        rospy.spin()
        
    except rospy.ROSInterruptException:
        # ROS中断异常，正常退出
        pass
    except Exception as e:
        # 其他异常，记录错误信息
        rospy.logerr(f"节点运行错误: {e}")

if __name__ == '__main__':
    main()