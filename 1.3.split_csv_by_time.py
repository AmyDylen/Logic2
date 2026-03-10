import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# ----------------------------
# 配置项
# ----------------------------
# 原始CSV文件所在目录
INPUT_DIR = "D:\malware-traffic-analysis\csv_MTA"  # 替换为你的原始CSV目录
# 切分后的文件输出目录
OUTPUT_DIR = "D:\malware-traffic-analysis\csv_MTA-1"
# 最大时间跨度（Timestamp差值）
MAX_TIME_SPAN = 3600000000
# 输出文件命名规则：原始文件名_批次号.csv（避免冲突）
SPLIT_SUFFIX = "_{:03d}.csv"

# ----------------------------
# 工具函数：创建输出目录
# ----------------------------
def create_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    # 移除子目录创建逻辑，所有文件直接放OUTPUT_DIR根目录

# ----------------------------
# 核心函数：切分单个CSV文件
# ----------------------------
def split_csv_file(csv_path):
    """
    切分单个CSV文件：
    1. 检查总时间跨度是否超过MAX_TIME_SPAN
    2. 若超过则按Session ID分组，保证同ID不拆分，切分为多个文件
    3. 每个切片的时间跨度 ≤ MAX_TIME_SPAN
    4. 保留原始子目录结构，切分文件保存在对应的子目录中
    """
    # 读取CSV文件
    try:
        df = pd.read_csv(csv_path, dtype={
            'Session ID': int,
            'Length': int,
            'Timestamp': float,
            'Session Duration': int,
            'Source IP': str,
            'Destination IP': str,
            'Source Port': int,
            'Destination Port': int,
            'Protocols': str,
            'Type': str,
            'Packet Count': int,
            'Block Count': int
        })
    except Exception as e:
        print(f"❌ 读取文件失败 {csv_path}: {e}")
        return

    # 空文件直接跳过
    if df.empty:
        print(f"⚠️ 文件为空 {csv_path}")
        return

    # 按Timestamp排序（确保时间顺序）
    df = df.sort_values('Timestamp').reset_index(drop=True)
    
    # 计算总时间跨度
    total_start = df['Timestamp'].min()
    total_end = df['Timestamp'].max()
    total_span = total_end - total_start

    # 提取原始文件的基础名称（用于生成输出文件名）
    file_name = os.path.basename(csv_path)
    file_base_name = os.path.splitext(file_name)[0]
    
    # 定义协议后缀列表
    protocol_suffixes = ['_tls1.3', '_tls1.2', '_tls1.1', '_tls1.0', '_ssl3.0', '_http', '_tcp']
    
    # 查找协议后缀位置，将批次号插入到协议后缀之前
    def generate_split_filename(base_name, batch_idx):
        """生成切分后的文件名，将批次号插入到协议后缀之前"""
        # 查找是否包含协议后缀
        for suffix in protocol_suffixes:
            if base_name.endswith(suffix):
                # 在协议后缀之前插入批次号
                name_without_suffix = base_name[:-len(suffix)]
                return f"{name_without_suffix}_{batch_idx:03d}{suffix}.csv"
        # 没有协议后缀，直接在末尾添加批次号
        return f"{base_name}_{batch_idx:03d}.csv"
    
    # 计算相对路径，保留子目录结构
    rel_path = os.path.relpath(os.path.dirname(csv_path), INPUT_DIR)
    if rel_path == '.':
        rel_path = ''
    
    # 构建输出子目录路径
    output_subdir = os.path.join(OUTPUT_DIR, rel_path)
    os.makedirs(output_subdir, exist_ok=True)

    # 时间跨度未超过阈值，直接保存到对应的子目录
    if total_span <= MAX_TIME_SPAN:
        output_path = os.path.join(output_subdir, file_name)
        df.to_csv(output_path, index=False)
        print(f"✅ 无需切分 {csv_path} → 保存至 {output_path}")
        return

    # ----------------------------
    # 需要切分的逻辑
    # ----------------------------
    print(f"🔪 需要切分 {csv_path} (总跨度: {total_span:.0f} > {MAX_TIME_SPAN})")
    
    # 1. 按Session ID分组，获取每个ID的时间范围
    session_time_info = {}
    for sid, group in df.groupby('Session ID'):
        sid_start = group['Timestamp'].min()
        sid_end = group['Timestamp'].max()
        session_time_info[sid] = {
            'start': sid_start,
            'end': sid_end,
            'data': group
        }
    
    # 2. 按Session ID的起始时间排序（保证切分顺序）
    sorted_sessions = sorted(session_time_info.items(), key=lambda x: x[1]['start'])
    
    # 3. 切分Session ID为多个批次，每个批次时间跨度 ≤ MAX_TIME_SPAN
    split_batches = []
    current_batch = []
    current_batch_start = None
    current_batch_end = 0

    for sid, info in sorted_sessions:
        sid_start = info['start']
        sid_end = info['end']
        
        # 初始化当前批次
        if not current_batch:
            current_batch_start = sid_start
            current_batch.append((sid, info))
            current_batch_end = sid_end
            continue
        
        # 检查加入当前批次后是否超过阈值
        new_batch_end = max(current_batch_end, sid_end)
        new_batch_span = new_batch_end - current_batch_start
        
        if new_batch_span <= MAX_TIME_SPAN:
            # 加入当前批次
            current_batch.append((sid, info))
            current_batch_end = new_batch_end
        else:
            # 保存当前批次，新建批次
            split_batches.append(current_batch)
            current_batch = [(sid, info)]
            current_batch_start = sid_start
            current_batch_end = sid_end
    
    # 加入最后一个批次
    if current_batch:
        split_batches.append(current_batch)
    
    # 4. 生成每个批次的CSV文件（保存到对应的子目录）
    for batch_idx, batch in enumerate(split_batches, 1):
        # 合并当前批次的所有Session ID数据
        batch_data = []
        batch_start = None
        batch_end = 0
        
        for sid, info in batch:
            batch_data.append(info['data'])
            if batch_start is None:
                batch_start = info['start']
            batch_end = max(batch_end, info['end'])
        
        batch_df = pd.concat(batch_data, ignore_index=True)
        # 按Timestamp重新排序
        batch_df = batch_df.sort_values('Timestamp').reset_index(drop=True)
        
        # 生成输出文件名：将批次号插入到协议后缀之前
        output_file_name = generate_split_filename(file_base_name, batch_idx)
        output_path = os.path.join(output_subdir, output_file_name)
        
        # 保存文件（到对应的子目录）
        batch_df.to_csv(output_path, index=False)
        
        # 打印日志
        batch_span = batch_end - batch_start
        print(f"  → 批次 {batch_idx}: 包含 {len(batch)} 个Session ID, 时间跨度 {batch_span:.0f}, 保存至 {output_path}")

# ----------------------------
# 遍历所有CSV文件并切分
# ----------------------------
def process_all_csv_files():
    # 创建输出目录（仅根目录）
    create_output_dir()
    
    # 获取所有CSV文件
    csv_files = []
    for root, _, files in os.walk(INPUT_DIR):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    if not csv_files:
        print(f"⚠️ 未在 {INPUT_DIR} 找到任何CSV文件")
        return
    
    # 逐个处理CSV文件
    print(f"\n📊 开始处理 {len(csv_files)} 个CSV文件...")
    for csv_path in tqdm(csv_files, desc="处理进度"):
        split_csv_file(csv_path)
    
    print(f"\n🎉 处理完成！所有切分文件保存在 {OUTPUT_DIR} (保留原始子目录结构)")

# ----------------------------
# 主函数
# ----------------------------
if __name__ == "__main__":
    # 配置检查
    if not os.path.exists(INPUT_DIR):
        print(f"❌ 输入目录不存在: {INPUT_DIR}")
    else:
        process_all_csv_files()