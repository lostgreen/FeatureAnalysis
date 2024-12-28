import os
import shutil
import librosa
from pydub import AudioSegment
import numpy as np
from openai import OpenAI
from tqdm import tqdm
import pandas as pd
import json
import pickle
import csv


def get_total_files(path):
    """递归计算路径下的所有文件数量"""
    total_files = 0
    for _, _, files in os.walk(path):
        total_files += len(files)
    return total_files
  

def get_volume_mark(file_name):
  def convert_mp3_to_wav(mp3_path):
    # 获取 MP3 文件的文件名（不包括路径）
    mp3_filename = os.path.basename(mp3_path)
    
    # print(mp3_filename)
    # 创建新的 WAV 文件名（在当前工作目录下）
    
    wav_filename = mp3_filename.replace('.mp3', '.wav').replace('.MP3', '.wav')
    
    #  print(wav_filename)
    # 如果文件已经在当前目录，则不需要复制
    
    if not os.path.exists(wav_filename):
        # 将 MP3 文件复制到当前工作目录
        shutil.copy2(mp3_path, mp3_filename)

    # 加载音频文件
    sound = AudioSegment.from_mp3(mp3_filename)

    # 导出为 WAV 格式
    wav_path = os.path.join(os.getcwd(), wav_filename)
    sound.export(wav_path, format="wav")
    
    if mp3_filename.endswith(".WAV"):
      return wav_path
    
    os.remove(mp3_filename)
    
    return wav_path

  # 加载音频文件
  mp3_file = file_name
  wav_file = convert_mp3_to_wav(mp3_file)
  # 使用 librosa 加载 wav 文件
  y, sr = librosa.load(wav_file)

  # 定义帧长和帧移
  frame_length = 2048  # 例如 2048 个样本点
  hop_length = 512     # 例如 512 个样本点

  # 计算每个帧的 RMS 值
  frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
  rms_values = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

  # 最大最小归一化
  min_rms = np.min(rms_values)
  max_rms = np.max(rms_values)
  normalized_rms = (rms_values - min_rms) / (max_rms - min_rms)

  # 将归一化后的 RMS 值乘以 100
  scaled_rms = normalized_rms * 100

  # 计算所有帧的平均值
  average_volume = np.mean(scaled_rms)

  # print(f'Average Volume Feature: {average_volume:.2f}')

  # 清理临时文件
  os.remove(wav_file)

  return ("声音音量", average_volume)


def get_WPM_mark(file_name):
  def parse_timestamp(timestamp):
    h, m, s_ms = timestamp.split(':')
    s, ms = s_ms.split(',')
    return int(h), int(m), int(s), int(ms)

  def total_milliseconds(h, m, s, ms):
    return h * 3600000 + m * 60000 + s * 1000 + ms

  def calculate_interval(start_timestamp, end_timestamp):
    h1, m1, s1, ms1 = parse_timestamp(start_timestamp)
    h2, m2, s2, ms2 = parse_timestamp(end_timestamp)

    start_ms = total_milliseconds(h1, m1, s1, ms1)
    end_ms = total_milliseconds(h2, m2, s2, ms2)

    interval_ms = end_ms - start_ms
    return interval_ms

  def calculate_wpm(text, duration_ms):
    word_count = len(text.strip())
    minutes = duration_ms / 60000.0
    wpm = word_count / minutes
    return wpm

  with open(file_name, 'r') as file:
    srt_content = file.read()

  # 解析SRT文件内容
  entries = srt_content.strip().split('\n\n')
  wpm_list = []

  for entry in entries:
      lines = entry.split('\n')
      if len(lines) >= 3:
          timestamps = lines[1]
          start_timestamp, end_timestamp = timestamps.split(' --> ')
          text = ' '.join(lines[2:])

          duration_ms = calculate_interval(start_timestamp, end_timestamp)
          wpm = calculate_wpm(text, duration_ms)
          wpm_list.append(wpm)
  avg_speed = sum(wpm_list) / len(wpm_list)

  return ("每分钟平均吐字", int(avg_speed))


def get_interaction_mark(text):
  client = OpenAI(api_key="sk-51d2985a4cd14eba9709a9bf31ad0930", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
  details = ["情感：从文字的情感，幽默性，情感流露等方面考察", "交流性：从提供建议，提问，自我表达等方面考察", "凝聚力：从称呼、社交性话语使用等方面考察"]
  response = client.chat.completions.create(
      model="qwen-plus-1127",
      messages=[
          {"role": "system", "content": f"你是一个辩论手，需要根据用户提供的辩论稿,进行社会临场感打分，综合情感，交流性，凝聚力三种标准， {details}, 仅仅回复0-10之间的一个数字,不要回复分析过程。"},
          {"role": "user", "content": text},
      ],
      stream=False
    )
  
  # print(response.choices[0].message.content)
  return [("社会临场感", response.choices[0].message.content)]


def get_quality_mark(text):
  client = OpenAI(api_key="sk-51d2985a4cd14eba9709a9bf31ad0930", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
  standards = ["相关性标准", "可接受性标准", "充分性标准"]
  judge_result = []
  standard_details = ["如果一个论点的所有前提都支持该主张的真实性(或虚假性)，则该论点满足相关性标准", "如果一个论点的前提代表了无可争议的常识或事实，那么它就符合可接受性标准", "如果一个论证的前提提供了足够的证据来接受或拒绝该主张，则该论点就符合充分性标准"]
  for standard, standard_detail in zip(standards, standard_details):
    response = client.chat.completions.create(
      model="qwen-plus-1127",
      messages=[
          {"role": "system", "content": f"你是一个辩论手，需要根据用户提供的辩论稿，对论点论据进行{standard}打分，综合所有论点论据的分数，形成整个文段的打分, {standard_detail}.只需要回复0-10的数字,不要回复分析过程"},
          {"role": "user", "content": text},
      ],
      stream=False
    )
    # print(standard, response.choices[0].message.content)
    judge_result.append((standard, response.choices[0].message.content))
  return judge_result


def get_adu(text):
  client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key="sk-51d2985a4cd14eba9709a9bf31ad0930",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
  )

  prompt= "你是一个辩论手，需要把用户提供的辩论稿，根据以下定义，Claim（立论）：辩手在一段发言中的核心论点，表明立场并为自己的发言做一个整体上的总结；Premise（陈述）：辩手对自己提出的论点做补充说明，辩手在表明自己立场之后，会用一段陈述来解释。它们是推理过程中的起点，提供了得出特定结论的基础。在论证中，前提可以是事实、假设、定义或任何被认为是真实并用来支持进一步推理的陈述；Pieces of Evidence（证据片段）: 证据片段是用于支持或证明某个主张的具体事实、数据、引用、统计数字、案例研究或其他形式的实证材料。这些证据片段在论证中充当支持性角色，为前提提供了可观察或可验证的依据，增强了论证的可信度。将辩论稿按照以上单元进行划分，注意不要修改原文,不要加入自己的总结。"
  response = client.chat.completions.create(
      model="qwen-plus-latest",
      messages=[
          {"role": "system", "content": prompt},
          {"role": "user", "content": text},
      ],
      stream=False
  )
  
  return response.choices[0].message.content


def get_text_from_srt(file_path):
    # 打开并读取文件内容
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    text = []
    buffer = []

    for line in lines:
        # 去除行末的换行符
        clean_line = line.strip()

        # 如果当前行为空行，则认为一组字幕结束
        if not clean_line:
            if buffer:
                # 将缓存中的字幕文本添加到结果列表中
                text.append(' '.join(buffer))
                buffer = []  # 清空缓存
        else:
            if '0' <= clean_line[0] <= '9':
                continue
              
            buffer.append(clean_line)

    # 添加最后一组字幕（如果有的话）
    if buffer:
        text.append(' '.join(buffer))

    return '\n'.join(text)
    

from collections import defaultdict


path = "/public2/clw/奇葩说数据/奇葩说第2季字幕音频/"
season = "season2"
error_files = []
total = get_total_files(path)
pbar = tqdm(total=total, desc='Processing', unit='file')
final_feature = defaultdict(dict)

for root, dirs, files in os.walk(path):
    for idx, file in enumerate(files):
        pbar.update(1)
        # print(root.split("/")[-1])
        features = []
        file_path = os.path.join(root, file)
        if file.endswith(".WAV") or file.endswith(".MP3") or file.endswith(".wav") or file.endswith(".mp3"):
            volume= get_volume_mark(file_path)
            features.append(volume)
            final_feature[root.split("/")[-1]]["声音音量"] = volume
        else:
            features.append(get_WPM_mark(file_path))
            text = get_text_from_srt(file_path)
            final_feature[root.split("/")[-1]]["辩论原稿"] = text
            # print(text)
            try:
              interaction_mark = get_interaction_mark(text)
              features.append(interaction_mark)
              final_feature[root.split("/")[-1]]["社会临场感得分"] = interaction_mark
            except Exception as e:
              print(e)
              error_files.append((file, "interaction"))
            try:
              quality = get_quality_mark(text)
              features.append(quality)
              final_feature[root.split("/")[-1]]["辩论质量得分"] = quality
            except Exception as e:
              print(e)
              error_files.append((file, "quality"))
            try:
              adu = get_adu(text)
              features.append(adu)
              final_feature[root.split("/")[-1]]["最小辩论单元"] = adu
            except Exception as e:
              print(e)
              error_files.append((file, "adu"))
        # print(features)
        
        # final_feature[root.split("/")[-1]] = final_feature.get(root.split("/")[-1], []) + features
        with open(f"./{season}/{root.split("/")[-1]}.txt", "a") as f:
            for feature in features:
                f.write(str(feature) + "\n")

# 指定要保存的文件路径
output_file = 'final_feature.pkl'

# 使用 with 语句打开文件，并使用 pickle.dump() 方法将数据写入文件
with open(output_file, 'wb') as file:
    pickle.dump(final_feature, file)

print(f"Data has been saved to {output_file}")

# 收集所有的列名
column_names = set()
for inner_dict in final_feature.values():
    column_names.update(inner_dict.keys())
# 添加'name'列
column_names = ['name'] + sorted(column_names)

# 准备数据行
rows = []
for name, inner_dict in final_feature.items():
    row = [name] + [inner_dict.get(col, '') for col in column_names[1:]]
    rows.append(row)

# 写入CSV文件
output_file = 'output.csv'
with open(output_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # 写入表头
    writer.writerow(column_names)
    # 写入数据行
    writer.writerows(rows)

print(f"Data has been written to {output_file}")

with open(f"./{season}/error.txt", "w") as f:
  for x in error_files:
    f.write(str(x)+"\n")
            