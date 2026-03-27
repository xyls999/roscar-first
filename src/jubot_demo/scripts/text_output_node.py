#!/usr/bin/env python3
# coding=utf-8
# 文本输出节点
# 功能：接收并输出检测结果文本信息，支持批量文本输出

import rospy
from std_msgs.msg import String
import time

class TextOutputNode:
    """文本输出节点类"""
    def __init__(self):
        # 初始化文本输出节点
        rospy.init_node('text_output_node', anonymous=True)
        
        # 存储接收到的句子，用于批量输出
        self.words_list = []
        
        # 标志位，控制输出状态，防止重复输出
        self.should_output = False
        
        # 订阅/words话题，接收来自检测节点的文字消息
        rospy.Subscriber('/words', String, self.words_callback)
        
        # 订阅/bigin_words话题，接收输出触发信号
        rospy.Subscriber('/bigin_words', String, self.begin_callback)
        
        rospy.loginfo("文本输出节点已启动，等待接收消息...")
        rospy.loginfo("请先通过/words话题发送文字，然后通过/bigin_words话题触发输出")
        
    def words_callback(self, msg):
        """/words话题的回调函数"""
        # 功能：接收并存储来自检测节点的文字消息
        if self.should_output:
            rospy.logwarn("已经开始输出，忽略新的文字消息")
            return
            
        text = msg.data.strip()
        if len(text) > 0:
            self.words_list.append(text)
            rospy.loginfo(f"接收到文字消息: {text}")
            rospy.loginfo(f"当前已存储 {len(self.words_list)} 条消息")
        else:
            rospy.logwarn("接收到空消息，已忽略")
    
    def begin_callback(self, msg):
        """/bigin_words话题的回调函数"""
        # 功能：响应输出触发信号，开始批量输出存储的文字消息
        if self.should_output:
            rospy.logwarn("已经开始输出，忽略重复的开始信号")
            return
            
        if len(self.words_list) == 0:
            rospy.logwarn("没有可输出的文字内容，请先通过/words话题发送文字")
            return
            
        self.should_output = True
        rospy.loginfo("接收到开始输出信号，开始逐行输出文字...")
        
        # 开始输出
        self.output_sentences()
    
    def output_sentences(self):
        """逐行输出所有存储的句子"""
        rospy.loginfo(f"开始输出 {len(self.words_list)} 条消息")
        rospy.loginfo("-" * 50)
        
        for i, sentence in enumerate(self.words_list, 1):
            rospy.loginfo(f"第 {i} 句: {sentence}")
            
            # 在控制台输出文字
            # print(f"第{i}行: {sentence}")
            
            # 在句子之间添加短暂停顿（除了最后一个句子）
            if i < len(self.words_list):
                time.sleep(1)
        
        rospy.loginfo("-" * 50)
        rospy.loginfo("所有消息输出完毕！")
        
        # 可选：输出完成后关闭节点
        rospy.loginfo("文字输出完成，节点将继续运行等待新消息")
        # 如果希望输出完成后关闭节点，可以取消下面的注释
        # rospy.signal_shutdown("文字输出完成")
    
    def run(self):
        """运行节点"""
        rospy.spin()  # 保持节点运行，等待消息

def main():
    """主函数"""
    try:
        output_node = TextOutputNode()
        output_node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("节点被中断")
    except Exception as e:
        rospy.logerr(f"节点运行出错: {e}")

if __name__ == "__main__":
    main()