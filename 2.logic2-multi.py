"""
LOGIC2 多协议统一脚本
支持TLS、HTTP、TCP三种协议，自动检测协议类型并加载对应预训练模型
"""

import os
import sys
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import argparse
from collections import defaultdict
import copy

# ==================== 配置日志 ====================

class Logger:
    """自定义日志记录器"""
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(log_dir, f'logic2_multi_{timestamp}.log')
        self.file_handle = open(self.log_file, 'w', encoding='utf-8')
        self._original_print = print
        self._original_print(f"日志文件: {self.log_file}")
        self.write(f"=== LOGIC2 Multi-Protocol ===")
        self.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.write("=" * 50)

    def write(self, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_line = f"[{timestamp}] {message}"
        self.file_handle.write(log_line + '\n')
        self.file_handle.flush()
        self._original_print(message)

    def close(self):
        self.write(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.file_handle.close()

logger = Logger()

def custom_print(*args, **kwargs):
    message = ' '.join(str(arg) for arg in args)
    logger.write(message)

print = custom_print

# ==================== 协议类型检测 ====================

def detect_protocol(filename):
    """检测文件名对应的协议类型"""
    # TLS协议检测
    tls_patterns = ['_tls1.0.csv', '_tls1.1.csv', '_tls1.2.csv', '_tls1.3.csv', '_ssl3.0.csv']
    for pattern in tls_patterns:
        if filename.endswith(pattern):
            return 'TLS'
    
    # HTTP协议检测
    if filename.endswith('_http.csv'):
        return 'HTTP'
    
    # TCP协议检测
    if filename.endswith('_tcp.csv'):
        return 'TCP'
    
    return 'Unknown'

# ==================== 基础数据结构 ====================

class FlowBlock:
    """表示单次数据传输"""
    def __init__(self, direction, payload_len, time_interval, timestamp):
        self.direction = direction
        self.payload_len = payload_len
        self.time_interval = time_interval
        self.timestamp = timestamp

class Flow:
    """表示单个流"""
    def __init__(self, session_id, src_ip, dst_ip, src_port, dst_port, protocol):
        self.session_id = session_id
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.src_port = src_port
        self.dst_port = dst_port
        self.protocol = protocol
        self.blocks = []
        self.duration = 0
        self.is_long_flow = False

class FlowCluster:
    """表示流簇"""
    def __init__(self, src_ip, dst_ip, dst_port, protocol, filename=None):
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.dst_port = dst_port
        self.protocol = protocol
        self.filename = filename  # 存储CSV文件名
        self.flows = []
        self.logical_sequence = []
        self.graph = None

# ==================== 多协议数据集 ====================

class MultiProtocolDataset(Dataset):
    """多协议数据集 - 自动按协议分类
    
    筛选规则（filter_mode参数）：
    - 'multi_only': 只保留多流簇（流数量 > 1），过滤掉所有单流簇
    - 'multi_and_long_single': 保留多流簇 + 单流长流（单流但is_long_flow=True）
    - 'all': 保留所有流簇（不过滤）
    """
    def __init__(self, csv_folder, is_pretrain=True, max_flows_per_cluster=100, filter_mode='multi_only'):
        self.csv_folder = csv_folder
        self.is_pretrain = is_pretrain
        self.max_flows_per_cluster = max_flows_per_cluster
        self.filter_mode = filter_mode
        
        # 按协议分类存储文件
        self.protocol_files = {
            'TLS': [],
            'HTTP': [],
            'TCP': [],
            'Unknown': []
        }
        
        # 递归扫描所有CSV文件
        for root, dirs, files in os.walk(csv_folder):
            for file in files:
                if file.endswith('.csv'):
                    protocol = detect_protocol(file)
                    rel_path = os.path.relpath(os.path.join(root, file), csv_folder)
                    self.protocol_files[protocol].append(rel_path)
        
        # 合并所有文件
        self.all_files = []
        self.file_protocols = []  # 记录每个文件的协议类型
        for protocol in ['TLS', 'HTTP', 'TCP']:
            for file in self.protocol_files[protocol]:
                self.all_files.append(file)
                self.file_protocols.append(protocol)

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            results = []
            for i in range(*idx.indices(len(self.all_files))):
                result = self._load_csv_to_flow_cluster(self.all_files[i], self.file_protocols[i])
                if result is not None:
                    results.append(result)
            return results
        else:
            csv_file = self.all_files[idx]
            protocol = self.file_protocols[idx]
            return self._load_csv_to_flow_cluster(csv_file, protocol)

    def _load_csv_to_flow_cluster(self, csv_file, protocol):
        with open(os.path.join(self.csv_folder, csv_file), 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            return None

        first_row = rows[0]
        src_ip = first_row['Source IP']
        dst_ip = first_row['Destination IP']
        dst_port = int(first_row['Destination Port'])

        # 提取文件名（不含路径）
        filename = os.path.basename(csv_file)
        flow_cluster = FlowCluster(src_ip, dst_ip, dst_port, protocol, filename)

        session_dict = {}
        for row in rows:
            session_id = int(row['Session ID'])
            if session_id not in session_dict:
                session_dict[session_id] = []
            session_dict[session_id].append(row)

        flow_count = 0
        for session_id, session_rows in session_dict.items():
            if flow_count >= self.max_flows_per_cluster:
                break

            first_session_row = session_rows[0]
            src_port = int(first_session_row['Source Port'])

            flow = Flow(session_id, src_ip, dst_ip, src_port, dst_port, protocol)

            prev_timestamp = None
            for row in session_rows:
                payload_len = int(float(row['Length']))
                direction = 0 if payload_len > 0 else 1
                payload_len = abs(payload_len)
                timestamp = int(float(row['Timestamp']))

                if prev_timestamp is None:
                    time_interval = 0
                else:
                    time_interval = timestamp - prev_timestamp

                block = FlowBlock(direction, payload_len, time_interval, timestamp)
                flow.blocks.append(block)
                prev_timestamp = timestamp

            flow.duration = int(float(session_rows[0]['Session Duration']))
            flow.is_long_flow = (len(flow.blocks) > 4) and (flow.duration > 3000000)

            flow_cluster.flows.append(flow)
            flow_count += 1

        # 根据筛选模式决定是否保留该流簇
        flow_count = len(flow_cluster.flows)
        
        if self.filter_mode == 'multi_only':
            # 只保留多流簇（流数量 > 1）
            if flow_count <= 1:
                return None
                
        elif self.filter_mode == 'multi_and_long_single':
            # 保留多流簇 + 单流长流
            if flow_count == 1:
                # 单流情况：只保留长流
                if not flow_cluster.flows[0].is_long_flow:
                    return None
            # flow_count > 1 的多流簇都保留
            
        elif self.filter_mode == 'all':
            # 保留所有流簇（不过滤）
            pass
        
        flow_cluster.logical_sequence = self._build_logical_sequence(flow_cluster)
        return flow_cluster

    def _build_logical_sequence(self, flow_cluster):
        all_blocks = []
        for flow in flow_cluster.flows:
            for block in flow.blocks:
                all_blocks.append((block.timestamp, block))
        all_blocks.sort(key=lambda x: x[0])
        return [block for _, block in all_blocks]

# ==================== 从原脚本导入模型组件 ====================

# 动态导入原脚本中的模型定义，确保与预训练模型兼容
# 注意：导入前保存当前的print函数，防止被覆盖
_multi_print = print  # 保存multi脚本的print函数

import importlib.util
spec = importlib.util.spec_from_file_location("logic2_tls", "2.logic2-tls.py")
logic2_module = importlib.util.module_from_spec(spec)
sys.modules['logic2_tls_module'] = logic2_module
spec.loader.exec_module(logic2_module)

# 恢复multi脚本的print函数（防止被tls脚本覆盖）
print = _multi_print

# 获取模型类
SequenceEncoder = logic2_module.SequenceEncoder
GraphEncoder = logic2_module.GraphEncoder
Logic2Model = logic2_module.Logic2Model
_build_flow_graph = logic2_module._build_flow_graph

# 导入样本增强模块
try:
    from flow_cluster_augmentation import FlowClusterAugmentation
    AUGMENTATION_AVAILABLE = True
except ImportError:
    AUGMENTATION_AVAILABLE = False
    print("⚠️  未找到样本增强模块 flow_cluster_augmentation.py")


def apply_augmentation(data, target_ratio=0.5):
    """对训练数据应用样本增强，使少数类达到目标比例
    
    Args:
        data: 训练数据列表 [(flow_cluster, label), ...]
        target_ratio: 目标少数类比例（默认0.5表示少数类占50%）
    
    Returns:
        增强后的数据列表
    """
    if not AUGMENTATION_AVAILABLE or len(data) == 0:
        return data
    
    augmenter = FlowClusterAugmentation()
    
    # 分离C2和良性样本
    c2_data = [(fc, label) for fc, label in data if label == 1.0]
    benign_data = [(fc, label) for fc, label in data if label == 0.0]
    
    # 确定少数类和多数类
    if len(c2_data) < len(benign_data):
        minority_data = c2_data
        majority_data = benign_data
        minority_class = "C2"
        minority_label = 1.0
    else:
        minority_data = benign_data
        majority_data = c2_data
        minority_class = "良性"
        minority_label = 0.0
    
    # 计算需要的增强数量
    # 目标: 少数类 / (少数类 + 多数类) = target_ratio
    # 设需要增强到 n 个少数类样本
    # n / (n + len(majority_data)) = target_ratio
    # n = target_ratio * (n + len(majority_data))
    # n = target_ratio * n + target_ratio * len(majority_data)
    # n * (1 - target_ratio) = target_ratio * len(majority_data)
    # n = target_ratio * len(majority_data) / (1 - target_ratio)
    
    if len(minority_data) > 0 and target_ratio < 1.0:
        target_minority_count = int(target_ratio * len(majority_data) / (1 - target_ratio))
        target_minority_count = max(target_minority_count, len(minority_data))  # 至少保留原有数量
        
        # 计算需要增强的次数
        additional_needed = target_minority_count - len(minority_data)
        
        print(f"  样本增强: 少数类={minority_class}")
        print(f"    原始: C2={len(c2_data)}, 良性={len(benign_data)}, 总计={len(data)}")
        print(f"    目标: 少数类占比={target_ratio:.1%}, 需要{target_minority_count}个少数类样本")
        print(f"    需增强: {additional_needed}个样本")
    else:
        print(f"  样本增强: 无需增强")
        return data
    
    # 增强后的数据
    augmented_data = majority_data.copy()
    augmented_data.extend(minority_data)  # 保留原始少数类样本
    
    # 对少数类样本进行增强，直到达到目标数量
    generated_count = 0
    while generated_count < additional_needed and len(minority_data) > 0:
        for flow_cluster, label in minority_data:
            if generated_count >= additional_needed:
                break
            augmented_cluster = augmenter.augment(flow_cluster)
            augmented_data.append((augmented_cluster, label))
            generated_count += 1
    
    # 打乱数据
    np.random.shuffle(augmented_data)
    
    # 统计增强后的分布
    final_c2 = sum(1 for _, label in augmented_data if label == 1.0)
    final_benign = sum(1 for _, label in augmented_data if label == 0.0)
    final_ratio = final_c2 / len(augmented_data) if len(augmented_data) > 0 else 0
    print(f"    增强后: C2={final_c2}, 良性={final_benign}, 总计={len(augmented_data)}, C2占比={final_ratio:.1%}")
    
    return augmented_data

# ==================== 多协议模型管理器 ====================

class MultiProtocolLogic2:
    """多协议LOGIC2管理器"""
    def __init__(self, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """加载各协议的预训练模型"""
        protocols = ['tls', 'http', 'tcp']
        
        for protocol in protocols:
            seq_path = f"./sequence_encoder_pretrained_{protocol}.pth"
            graph_path = f"./graph_encoder_pretrained_{protocol}.pth"
            
            sequence_encoder = SequenceEncoder()
            graph_encoder = GraphEncoder(output_dim=128)
            
            # 尝试加载预训练权重
            if os.path.exists(seq_path):
                try:
                    sequence_encoder.load_state_dict(torch.load(seq_path, weights_only=True))
                    print(f"✓ 加载{protocol.upper()}序列编码器: {seq_path}")
                except Exception as e:
                    print(f"⚠️  {protocol.upper()}序列编码器加载失败: {e}")
            else:
                print(f"⚠️  未找到{protocol.upper()}序列编码器: {seq_path}")
            
            if os.path.exists(graph_path):
                try:
                    graph_encoder.load_state_dict(torch.load(graph_path, weights_only=True), strict=False)
                    print(f"✓ 加载{protocol.upper()}图编码器: {graph_path}")
                except Exception as e:
                    print(f"⚠️  {protocol.upper()}图编码器加载失败: {e}")
            else:
                print(f"⚠️  未找到{protocol.upper()}图编码器: {graph_path}")
            
            model = Logic2Model(sequence_encoder, graph_encoder)
            model.to(self.device)
            self.models[protocol.upper()] = model
    
    def predict(self, flow_cluster):
        """预测单个流簇"""
        protocol = flow_cluster.protocol
        if protocol not in self.models:
            print(f"⚠️  未知协议: {protocol}，使用TLS模型")
            protocol = 'TLS'
        
        model = self.models[protocol]
        model.eval()
        
        with torch.no_grad():
            pred = model(flow_cluster)
            prob = torch.sigmoid(pred).item()
        
        return prob

# ==================== 评估函数 ====================

def evaluate_multi_protocol(data, multi_protocol_model):
    """多协议评估"""
    import time
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_protocols = []
    misclassified = []
    single_flow_short_count = 0
    
    protocol_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0})
    
    # 记录检测开始时间
    start_time = time.time()
    processed_count = 0
    
    for i, (flow_cluster, label) in enumerate(data):
        protocol = flow_cluster.protocol
        
        # 检查是否为单流短流
        is_single_short = (len(flow_cluster.flows) == 1 and not flow_cluster.flows[0].is_long_flow)
        if is_single_short:
            single_flow_short_count += 1
            pred_prob = 0.01
            pred_label = 0.0
        else:
            pred_prob = multi_protocol_model.predict(flow_cluster)
            pred_label = 1.0 if pred_prob > 0.5 else 0.0
        
        # 统计
        all_preds.append(pred_label)
        all_labels.append(label)
        all_probs.append(pred_prob)
        all_protocols.append(protocol)
        
        # 按协议统计
        protocol_stats[protocol]['total'] += 1
        if pred_label == label:
            protocol_stats[protocol]['correct'] += 1
        
        if label == 1.0 and pred_label == 1.0:
            protocol_stats[protocol]['tp'] += 1
        elif label == 0.0 and pred_label == 1.0:
            protocol_stats[protocol]['fp'] += 1
        elif label == 1.0 and pred_label == 0.0:
            protocol_stats[protocol]['fn'] += 1
        elif label == 0.0 and pred_label == 0.0:
            protocol_stats[protocol]['tn'] += 1
        
        # 记录错误分类
        if pred_label != label:
            filename = flow_cluster.filename if hasattr(flow_cluster, 'filename') and flow_cluster.filename else 'unknown'
            misclassified.append({
                'index': i,
                'protocol': protocol,
                'true_label': 'C2' if label == 1.0 else '良性',
                'pred_label': 'C2' if pred_label == 1.0 else '良性',
                'pred_prob': pred_prob,
                'flow_cluster_info': f"{flow_cluster.src_ip} → {flow_cluster.dst_ip}:{flow_cluster.dst_port}",
                'flow_count': len(flow_cluster.flows),
                'filename': filename
            })
        
        processed_count += 1
    
    # 计算检测速度
    end_time = time.time()
    total_time = end_time - start_time
    speed = processed_count / total_time if total_time > 0 else 0
    
    print(f"\n【检测速度统计】")
    print(f"  处理流簇总数: {processed_count}")
    print(f"  总耗时: {total_time:.2f} 秒")
    print(f"  平均速度: {speed:.2f} 流簇/秒")
    
    # 计算总体指标
    accuracy = sum(1 for p, l in zip(all_preds, all_labels) if p == l) / len(all_labels)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    roc_auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'misclassified': misclassified,
        'single_flow_short_count': single_flow_short_count,
        'protocol_stats': dict(protocol_stats),
        'all_protocols': all_protocols,
        'speed_stats': {
            'total_clusters': processed_count,
            'total_time_seconds': total_time,
            'clusters_per_second': speed
        }
    }

# ==================== 主函数 ====================

def train_multi_protocol(train_data, test_data, epochs=20, batch_size=16, lr=1e-4, warmup_epochs=5):
    """多协议联合训练"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 为每个协议创建模型
    protocols = ['TLS', 'HTTP', 'TCP']
    models = {}
    optimizers = {}
    
    for protocol in protocols:
        sequence_encoder = SequenceEncoder()
        graph_encoder = GraphEncoder(output_dim=128)
        model = Logic2Model(sequence_encoder, graph_encoder)
        model.to(device)
        models[protocol] = model
        optimizers[protocol] = optim.Adam(model.parameters(), lr=lr)
    
    # 按协议分组数据
    protocol_train_data = {p: [] for p in protocols}
    protocol_test_data = {p: [] for p in protocols}
    
    for fc, label in train_data:
        if fc.protocol in protocol_train_data:
            protocol_train_data[fc.protocol].append((fc, label))
    
    for fc, label in test_data:
        if fc.protocol in protocol_test_data:
            protocol_test_data[fc.protocol].append((fc, label))
    
    # 打印各协议数据分布（增强前）
    print("\n各协议训练数据分布（增强前）:")
    for protocol in protocols:
        c2_count = sum(1 for _, label in protocol_train_data[protocol] if label == 1.0)
        benign_count = sum(1 for _, label in protocol_train_data[protocol] if label == 0.0)
        print(f"  {protocol}: C2={c2_count}, 良性={benign_count}, 总计={len(protocol_train_data[protocol])}")
    
    # 应用样本增强
    print("\n" + "=" * 80)
    print("样本增强 (目标: C2占比50%，平衡数据集)")
    print("=" * 80)

    for protocol in protocols:
        if len(protocol_train_data[protocol]) > 0:
            protocol_train_data[protocol] = apply_augmentation(protocol_train_data[protocol], target_ratio=0.5)
    
    best_val_f1 = {p: 0.0 for p in protocols}
    patience_counter = {p: 0 for p in protocols}
    patience = 5
    
    # 阶段一：Warmup（冻结编码器，只训练判别器）
    print("\n" + "=" * 80)
    print("阶段一：判别器预热 (Warmup)")
    print("=" * 80)
    
    for epoch in range(warmup_epochs):
        print(f"\nWarmup Epoch {epoch+1}/{warmup_epochs}")
        
        for protocol in protocols:
            if len(protocol_train_data[protocol]) == 0:
                continue
            
            model = models[protocol]
            optimizer = optimizers[protocol]
            
            # 冻结编码器
            for param in model.sequence_encoder.parameters():
                param.requires_grad = False
            for param in model.graph_encoder.parameters():
                param.requires_grad = False
            for param in model.instance_mlp.parameters():
                param.requires_grad = True
            
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            np.random.shuffle(protocol_train_data[protocol])
            
            for i in range(0, len(protocol_train_data[protocol]), batch_size):
                batch = protocol_train_data[protocol][i:i+batch_size]
                
                optimizer.zero_grad()
                batch_loss = 0
                
                for fc, label in batch:
                    pred = model(fc)
                    if pred.dim() == 0:
                        pred = pred.unsqueeze(0)
                    label_tensor = torch.tensor([label], device=device)
                    loss = nn.BCEWithLogitsLoss()(pred, label_tensor)
                    batch_loss += loss
                    
                    prob = torch.sigmoid(pred).item()
                    pred_label = 1.0 if prob > 0.5 else 0.0
                    if pred_label == label:
                        correct += 1
                    total += 1
                
                if len(batch) > 0:
                    batch_loss = batch_loss / len(batch)
                    batch_loss.backward()
                    optimizer.step()
                    total_loss += batch_loss.item()
            
            train_acc = correct / total if total > 0 else 0
            print(f"  {protocol}: Loss={total_loss/(len(protocol_train_data[protocol])/batch_size+1):.4f}, Acc={train_acc:.4f}")
    
    # 阶段二：Finetune（解冻联合微调）
    print("\n" + "=" * 80)
    print("阶段二：受控解冻联合微调 (Finetune)")
    print("=" * 80)
    
    for epoch in range(epochs):
        print(f"\nFinetune Epoch {epoch+1}/{epochs}")
        
        for protocol in protocols:
            if len(protocol_train_data[protocol]) == 0:
                continue
            
            model = models[protocol]
            optimizer = optimizers[protocol]
            
            # 解冻所有参数
            for param in model.parameters():
                param.requires_grad = True
            
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            np.random.shuffle(protocol_train_data[protocol])
            
            for i in range(0, len(protocol_train_data[protocol]), batch_size):
                batch = protocol_train_data[protocol][i:i+batch_size]
                
                optimizer.zero_grad()
                batch_loss = 0
                
                for fc, label in batch:
                    pred = model(fc)
                    if pred.dim() == 0:
                        pred = pred.unsqueeze(0)
                    label_tensor = torch.tensor([label], device=device)
                    loss = nn.BCEWithLogitsLoss()(pred, label_tensor)
                    batch_loss += loss
                    
                    prob = torch.sigmoid(pred).item()
                    pred_label = 1.0 if prob > 0.5 else 0.0
                    if pred_label == label:
                        correct += 1
                    total += 1
                
                if len(batch) > 0:
                    batch_loss = batch_loss / len(batch)
                    batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    total_loss += batch_loss.item()
            
            train_acc = correct / total if total > 0 else 0
            
            # 验证
            if len(protocol_test_data[protocol]) > 0:
                model.eval()
                val_preds = []
                val_labels = []
                
                with torch.no_grad():
                    for fc, label in protocol_test_data[protocol]:
                        pred = model(fc)
                        prob = torch.sigmoid(pred).item()
                        pred_label = 1.0 if prob > 0.5 else 0.0
                        val_preds.append(pred_label)
                        val_labels.append(label)
                
                val_f1 = f1_score(val_labels, val_preds, zero_division=0)
                print(f"  {protocol}: Loss={total_loss/(len(protocol_train_data[protocol])/batch_size+1):.4f}, Train Acc={train_acc:.4f}, Val F1={val_f1:.4f}")
                
                # Early stopping
                if val_f1 > best_val_f1[protocol]:
                    best_val_f1[protocol] = val_f1
                    patience_counter[protocol] = 0
                    # 保存最佳模型
                    torch.save(model.state_dict(), f"./logic2_model_best_{protocol.lower()}.pth")
                else:
                    patience_counter[protocol] += 1
            else:
                print(f"  {protocol}: Loss={total_loss/(len(protocol_train_data[protocol])/batch_size+1):.4f}, Train Acc={train_acc:.4f}")
        
        # 检查是否所有协议都早停
        if all(patience_counter[p] >= patience for p in protocols if len(protocol_train_data[p]) > 0):
            print(f"\n所有协议触发早停，停止训练")
            break
    
    # 加载最佳模型
    print("\n加载各协议最佳模型...")
    for protocol in protocols:
        model_path = f"./logic2_model_best_{protocol.lower()}.pth"
        if os.path.exists(model_path) and len(protocol_train_data[protocol]) > 0:
            models[protocol].load_state_dict(torch.load(model_path, weights_only=True))
            print(f"  ✓ {protocol}: {model_path}")
    
    return models


def main():
    parser = argparse.ArgumentParser(description="LOGIC2 多协议统一检测")
    parser.add_argument("--c2-data", type=str, required=True, help="C2数据文件夹路径")
    parser.add_argument("--benign-data", type=str, required=True, help="良性数据文件夹路径")
    parser.add_argument("--test-split", type=float, default=0.2, help="测试集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--mode", type=str, default="eval", choices=["train", "eval"], 
                        help="运行模式: train=训练+评估, eval=仅评估")
    parser.add_argument("--epochs", type=int, default=20, help="微调轮数")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="预热轮数")
    parser.add_argument("--batch-size", type=int, default=16, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--filter-mode", type=str, default="multi_only", 
                        choices=["multi_only", "multi_and_long_single", "all"],
                        help="数据筛选模式: multi_only=只保留多流簇, "
                             "multi_and_long_single=保留多流簇+单流长流, all=保留所有")
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("=" * 80)
    print("LOGIC2 多协议检测")
    print("=" * 80)
    
    # 加载数据（根据筛选模式过滤）
    filter_mode_desc = {
        'multi_only': '只保留多流簇',
        'multi_and_long_single': '保留多流簇+单流长流',
        'all': '保留所有流簇'
    }
    print(f"\n数据筛选模式: {args.filter_mode} ({filter_mode_desc.get(args.filter_mode, '')})")
    
    print(f"\n加载C2数据: {args.c2_data}")
    c2_dataset = MultiProtocolDataset(args.c2_data, is_pretrain=False, filter_mode=args.filter_mode)
    
    # 加载所有C2样本
    c2_data_all = []
    c2_protocol_counts = defaultdict(int)
    for idx in range(len(c2_dataset)):
        fc = c2_dataset[idx]
        if fc is not None:  # 根据filter_mode过滤
            c2_data_all.append((fc, 1.0))
            c2_protocol_counts[fc.protocol] += 1
    
    print(f"  C2样本总数（过滤后）: {len(c2_data_all)}")
    print("  协议分布:")
    for protocol, count in sorted(c2_protocol_counts.items()):
        print(f"    {protocol}: {count}")
    
    print(f"\n加载良性数据: {args.benign_data}")
    benign_dataset = MultiProtocolDataset(args.benign_data, is_pretrain=False, filter_mode=args.filter_mode)
    
    # 加载所有良性样本
    benign_data_all = []
    benign_protocol_counts = defaultdict(int)
    for idx in range(len(benign_dataset)):
        fc = benign_dataset[idx]
        if fc is not None:  # 根据filter_mode过滤
            benign_data_all.append((fc, 0.0))
            benign_protocol_counts[fc.protocol] += 1
    
    print(f"  良性样本总数（过滤后）: {len(benign_data_all)}")
    print("  协议分布:")
    for protocol, count in sorted(benign_protocol_counts.items()):
        print(f"    {protocol}: {count}")
    
    # 划分数据集
    print(f"\n划分数据集 (测试集比例: {args.test_split}, 随机种子: {args.seed})")
    
    # 在划分前设置随机种子，确保可复现
    np.random.seed(args.seed)
    
    # C2数据
    np.random.shuffle(c2_data_all)
    n_c2_test = int(len(c2_data_all) * args.test_split)
    c2_test_data = c2_data_all[:n_c2_test]
    c2_train_data = c2_data_all[n_c2_test:]
    
    # 良性数据
    np.random.shuffle(benign_data_all)
    n_benign_test = int(len(benign_data_all) * args.test_split)
    benign_test_data = benign_data_all[:n_benign_test]
    benign_train_data = benign_data_all[n_benign_test:]
    
    # 构建训练集和测试集
    train_data = c2_train_data + benign_train_data
    test_data = c2_test_data + benign_test_data
    
    np.random.shuffle(train_data)
    np.random.shuffle(test_data)
    
    print(f"  训练集: {len(train_data)} (C2: {len(c2_train_data)}, 良性: {len(benign_train_data)})")
    print(f"  测试集: {len(test_data)} (C2: {len(c2_test_data)}, 良性: {len(benign_test_data)})")
    
    if args.mode == "train":
        # 训练模式：训练 + 评估
        print("\n" + "=" * 80)
        print("【训练模式】先训练，后评估")
        print("=" * 80)
        models = train_multi_protocol(
            train_data, test_data,
            epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            batch_size=args.batch_size,
            lr=args.lr
        )
        
        # 创建多协议模型包装器用于评估
        class TrainedMultiProtocolModel:
            def __init__(self, models, device):
                self.models = models
                self.device = device
            
            def predict(self, flow_cluster):
                protocol = flow_cluster.protocol
                if protocol not in self.models:
                    protocol = 'TLS'
                model = self.models[protocol]
                model.eval()
                with torch.no_grad():
                    pred = model(flow_cluster)
                    prob = torch.sigmoid(pred).item()
                return prob
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        multi_model = TrainedMultiProtocolModel(models, device)
        
    else:
        # 评估模式 - 直接加载预训练模型进行评估
        print("\n" + "=" * 80)
        print("【评估模式】加载预训练模型进行评估")
        print("=" * 80)
        multi_model = MultiProtocolLogic2()
    
    # 评估测试集
    print("\n" + "=" * 80)
    print("评估测试集")
    print("=" * 80)
    
    results = evaluate_multi_protocol(test_data, multi_model)
    
    # 打印总体结果
    print("\n【总体结果】")
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1-Score:  {results['f1']:.4f}")
    print(f"  ROC-AUC:   {results['roc_auc']:.4f}")
    print(f"\n  混淆矩阵:")
    print(f"    {results['confusion_matrix']}")
    
    if results['single_flow_short_count'] > 0:
        print(f"\n  单流短流直接判定为良性: {results['single_flow_short_count']}个")
    
    # 打印各协议结果
    print("\n" + "=" * 80)
    print("各协议详细结果")
    print("=" * 80)
    
    for protocol, stats in sorted(results['protocol_stats'].items()):
        if stats['total'] == 0:
            continue
        
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        precision = stats['tp'] / (stats['tp'] + stats['fp']) if (stats['tp'] + stats['fp']) > 0 else 0
        recall = stats['tp'] / (stats['tp'] + stats['fn']) if (stats['tp'] + stats['fn']) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n【{protocol}】")
        print(f"  样本数: {stats['total']}")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  TP: {stats['tp']}, FP: {stats['fp']}, FN: {stats['fn']}, TN: {stats['tn']}")
    
    # 打印误报信息
    if results['misclassified']:
        print("\n" + "=" * 80)
        print(f"误报分析 (共{len(results['misclassified'])}个)")
        print("=" * 80)
        
        for i, item in enumerate(results['misclassified']):
            print(f"\n[{i+1}] {item['flow_cluster_info']}")
            print(f"    文件名: {item['filename']}")
            print(f"    协议: {item['protocol']}")
            print(f"    真实标签: {item['true_label']}, 预测标签: {item['pred_label']}")
            print(f"    预测概率: {item['pred_prob']:.4f}")
            print(f"    流数量: {item['flow_count']}")
    
    print("\n" + "=" * 80)
    print("评估完成")
    print("=" * 80)

if __name__ == "__main__":
    main()
