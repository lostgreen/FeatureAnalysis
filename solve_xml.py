import xml.etree.ElementTree as ET
from collections import defaultdict
from snownlp import SnowNLP
import os
from tqdm import tqdm
import json


def analyze_sentiment(text):
    s = SnowNLP(text)
    sentiment_score = s.sentiments  # 返回值在 [0, 1] 之间，越接近1表示越积极
    if sentiment_score > 0.6:
        return "Positive"
    elif sentiment_score < 0.4:
        return "Negative"
    else:
        return "Neutral"


def get_total_files(path):
    """递归计算路径下的所有文件数量"""
    total_files = 0
    for _, _, files in os.walk(path):
        total_files += len(files)
    return total_files


def solve_xml(xml_path):
    
    # 遍历每个bulletInfo元素并进行情感分析
    output_list = []
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    for entry in root.findall('.//entry'):
        for bullet_info in entry.find('.//list').findall('bulletInfo'):
            content = bullet_info.find('content').text
            show_time = bullet_info.find('showTime').text
            
            sentiment = analyze_sentiment(content)
            
            output_list.append((sentiment, show_time))
            
    return output_list
            



path = "/public2/clw/奇葩说数据/奇葩说第2季弹幕/"
total = get_total_files(path)
# print(total)
pbar = tqdm(total=total, desc='Processing', unit='file')
# final_feature = defaultdict(dict)
dump_path = "bullet_season2/"


for root, dirs, files in os.walk(path):
    # print(root.split("/")[-1])
    episode = []
    for idx, file in enumerate(files):
        pbar.update(1)
        file_path = os.path.join(root, file)
        if file_path.endswith(".xml"):
            episode += solve_xml(file_path)
            
    file_name = f"{dump_path}{os.path.basename(root)}.json"
    with open(file_name, 'w', encoding='utf-8') as file:
        json.dump(episode, file, ensure_ascii=False, indent=4)
