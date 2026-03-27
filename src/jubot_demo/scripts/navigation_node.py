#!/usr/bin/env python3
# coding=utf-8
# 导航节点 - 手动启动版本
# 功能：节点启动后立即开始导航任务

import rospy
import actionlib
import math
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from std_msgs.msg import Empty, String, Bool
import subprocess
import threading

# 航点字典 - 定义机器人导航的各个航点位置和拍照标记
WAYPOINTS_DICT = {
    1: {
        "Pos_x": -1.62,
        "Pos_y": 1.7,
        "Pos_z": 0.0,
        "Ori_x": -0.0,
        "Ori_y": 0.0,
        "Ori_z": -0.71,
        "Ori_w": 0.71,
        "take_photo": False
    },
    2: {
        "Pos_x": -1.62,
        "Pos_y": 0.5,
        "Pos_z": 0.0,
        "Ori_x": 0.0,
        "Ori_y": 0.0,
        "Ori_z": -0.71,
        "Ori_w": 0.71,
        "take_photo": False  
    },
    3: {
        "Pos_x": -0.8,
        "Pos_y": 0.6,
        "Pos_z": 0.0,
        "Ori_x": 0.0,
        "Ori_y": 0.0,
        "Ori_z": -0.71,
        "Ori_w": 0.71,
        "take_photo": True  
    },
    4: {
        "Pos_x": -0.76,
        "Pos_y": 0.36,
        "Pos_z": 0.0,
        "Ori_x": 0.0,
        "Ori_y": 0.0,
        "Ori_z": 0.71,
        "Ori_w": 0.71,
        "take_photo": True  
    },
    5: {
        "Pos_x": -0.14,
        "Pos_y": 0.5,
        "Pos_z": 0.0,
        "Ori_x": -0.0,
        "Ori_y": -0.0,
        "Ori_z": -0.71,
        "Ori_w": 0.71,
        "take_photo": False
    },
    6: {
        "Pos_x": -0.14,
        "Pos_y": -1.7,
        "Pos_z": 0.0,
        "Ori_x": 0.0,
        "Ori_y": 0.0,
        "Ori_z": 0.0,
        "Ori_w": 1.0,
        "take_photo": False
    },
    7: {
        "Pos_x": 1.0,
        "Pos_y": -1.7,
        "Pos_z": 0.0,
        "Ori_x": 0.0,
        "Ori_y": 0.0,
        "Ori_z": 0.71,
        "Ori_w": 0.71,
        "take_photo": False
    },
    8: {
        "Pos_x": 1.0,
        "Pos_y": -1.1,
        "Pos_z": 0.0,
        "Ori_x": 0.0,
        "Ori_y": 0.0,
        "Ori_z": 0.0,
        "Ori_w": 1.0,
        "take_photo": True  
    },
    9: {
        "Pos_x": 1.0,
        "Pos_y": 1.7,
        "Pos_z": 0.0,
        "Ori_x": 0.0,
        "Ori_y": 0.0,
        "Ori_z": 1.0,
        "Ori_w": 0.0,
        "take_photo": False
    },
    10: {
        "Pos_x": 1.7,
        "Pos_y": 1.7,
        "Pos_z": 0.0,
        "Ori_x": 0.0,
        "Ori_y": 0.0,
        "Ori_z": 1.0,
        "Ori_w": 0.0,
        "take_photo": False
    },
}


class WaypointNavigator:
    """航点导航器类"""
    def __init__(self):
        # 创建move_base动作客户端，用于发送导航目标
        self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        
        # 创建发布器：拍照指令和文本输出触发
        self.photo_trigger_pub = rospy.Publisher('/robot/take_photo', Empty, queue_size=1)
        self.begin_words_pub = rospy.Publisher('/bigin_words', String, queue_size=1)
        
        rospy.sleep(1)

        rospy.loginfo("正在等待 move_base action server...")
        # 等待move_base服务10秒，如果超时则报错并退出
        if not self.move_base_client.wait_for_server(rospy.Duration(10.0)):
            rospy.logerr("连接 move_base server 失败! 请确保 move_base 节点已正确启动。")
            rospy.signal_shutdown("无法连接到 move_base")
            return
        rospy.loginfo("成功连接到 move_base server!")

    def take_photo(self):
        """发布拍照指令并等待拍照完成"""
        rospy.loginfo("准备拍照...")
        # 发布拍照触发信号给图像捕获节点
        rospy.sleep(3) 
        self.photo_trigger_pub.publish(Empty())
        rospy.sleep(2)  # 等待拍照和图像处理完成
        rospy.loginfo("拍照完成")

    def calculate_quaternion_from_yaw(self, yaw):
        """从yaw角计算四元数"""
        # 功能：将欧拉角yaw转换为四元数表示，用于机器人朝向控制
        half_yaw = yaw * 0.5
        return {
            "x": 0.0,
            "y": 0.0,
            "z": math.sin(half_yaw),
            "w": math.cos(half_yaw)
        }

    def execute_navigation(self):
        """执行完整的导航巡航任务"""
        rospy.loginfo("开始自动巡航任务...")
        
        # 处理特殊航点的四元数转换（如果存在）
        if "1_5" in WAYPOINTS_DICT:
            yaw = WAYPOINTS_DICT["1_5"]["Ori_z"]  # 假设yaw存储在Ori_z中
            quat = self.calculate_quaternion_from_yaw(yaw)
            WAYPOINTS_DICT["1_5"]["Ori_x"] = quat["x"]
            WAYPOINTS_DICT["1_5"]["Ori_y"] = quat["y"] 
            WAYPOINTS_DICT["1_5"]["Ori_z"] = quat["z"]
            WAYPOINTS_DICT["1_5"]["Ori_w"] = quat["w"]
        
        # 定义航点访问顺序，按顺序执行巡航任务
        waypoint_keys = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        for waypoint_id in waypoint_keys:     
            point_data = WAYPOINTS_DICT[waypoint_id]
            rospy.loginfo(f"正在前往航点 {waypoint_id}")
            
            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = 'map'
            goal.target_pose.header.stamp = rospy.Time.now()
            
            goal.target_pose.pose.position.x = point_data["Pos_x"]
            goal.target_pose.pose.position.y = point_data["Pos_y"]
            goal.target_pose.pose.position.z = point_data["Pos_z"]
            goal.target_pose.pose.orientation.x = point_data["Ori_x"]
            goal.target_pose.pose.orientation.y = point_data["Ori_y"]
            goal.target_pose.pose.orientation.z = point_data["Ori_z"]
            goal.target_pose.pose.orientation.w = point_data["Ori_w"]
            
            self.move_base_client.send_goal(goal)
            self.move_base_client.wait_for_result() # 等待导航结果
            
            if self.move_base_client.get_state() == actionlib.GoalStatus.SUCCEEDED:
                rospy.loginfo(f"成功到达航点 {waypoint_id}!")
                
                # 检查是否需要拍照
                if point_data.get("take_photo", False):
                    rospy.loginfo(f"航点 {waypoint_id} 需要拍照")
                    self.take_photo()
                
                rospy.sleep(0.15)
                
            else:
                rospy.logerr(f"未能到达航点 {waypoint_id}，任务中止")
                return
        
        rospy.loginfo("所有航点巡航完毕！任务完成")
        
        rospy.loginfo("发送/bigin_words消息，触发文本输出...")
        begin_msg = String()
        begin_msg.data = "navigation_complete"
        self.begin_words_pub.publish(begin_msg)
        rospy.loginfo("已发送触发消息")

if __name__ == "__main__":
    try:
        # 初始化ROS节点
        rospy.init_node('navigation_node', anonymous=True)
        
        # --- 修改部分：移除了30秒倒计时 ---
        rospy.loginfo("="*50)
        rospy.loginfo("系统初始化完成，立即启动导航任务...")
        rospy.loginfo("="*50)
        
        # 创建导航器并执行任务
        navigator = WaypointNavigator()
        navigator.execute_navigation()
            
    except rospy.ROSInterruptException:
        rospy.loginfo("导航任务被中断")
    except Exception as e:
        rospy.logerr(f"程序运行出现异常: {e}")