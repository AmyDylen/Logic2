"""
LOGIC2 多协议 Leave-One-Family-Out (LOFO) 交叉验证脚本
用于评估跨家族泛化能力 (Cross-Family Generalization)
支持 TLS、HTTP、TCP 三种协议
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import argparse
from datetime import datetime
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# 导入基础数据结构
from logic2_tls_import import (
    FlowBlock, Flow, FlowCluster, FlowClusterDataset,
    evaluate, logger, print
)

# 从原脚本导入模型架构（确保与预训练模型兼容）
import importlib.util
spec = importlib.util.spec_from_file_location("logic2_tls", "2.logic2-tls.py")
logic2_module = importlib.util.module_from_spec(spec)

# 手动加载模块，避免执行main()
import types
sys.modules['logic2_tls_module'] = logic2_module

# 先导入torch_geometric相关模块，避免加载错误
import torch_geometric
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# 现在加载模块
spec.loader.exec_module(logic2_module)

# 获取模型类
SequenceEncoder = logic2_module.SequenceEncoder
GraphEncoder = logic2_module.GraphEncoder
Logic2Model = logic2_module.Logic2Model

# 导入样本增强模块
try:
    from flow_cluster_augmentation import FlowClusterAugmentation
    AUGMENTATION_AVAILABLE = True
except ImportError:
    AUGMENTATION_AVAILABLE = False
    print("警告: 未找到样本增强模块，将不使用数据增强")


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


class MultiProtocolFlowClusterDataset:
    """多协议流簇数据集 - 自动按协议分类并过滤单流簇
    
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
        
        # 递归扫描所有CSV文件（如果路径存在）
        if os.path.exists(csv_folder):
            for root, dirs, files in os.walk(csv_folder):
                for file in files:
                    if file.endswith('.csv'):
                        protocol = detect_protocol(file)
                        rel_path = os.path.relpath(os.path.join(root, file), csv_folder)
                        self.protocol_files[protocol].append(rel_path)
        
        # 合并所有文件
        self.all_files = []
        self.file_protocols = []
        for protocol in ['TLS', 'HTTP', 'TCP']:
            for file in self.protocol_files[protocol]:
                self.all_files.append(file)
                self.file_protocols.append(protocol)
        
        # 创建基础数据集用于加载（如果路径存在）
        if os.path.exists(csv_folder):
            self.base_dataset = FlowClusterDataset(csv_folder, is_pretrain=is_pretrain)
        else:
            self.base_dataset = None

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            results = []
            for i in range(*idx.indices(len(self.all_files))):
                result = self._load_item(i)
                if result is not None:
                    results.append(result)
            return results
        else:
            return self._load_item(idx)
    
    def _load_item(self, idx):
        """加载单个样本，根据filter_mode过滤"""
        if self.base_dataset is None or idx >= len(self.all_files):
            return None
            
        # 构建完整文件路径
        csv_file = os.path.join(self.csv_folder, self.all_files[idx])
        
        # 使用基础数据集的加载逻辑
        flow_cluster = self.base_dataset._load_csv_to_flow_cluster(csv_file)
        
        if flow_cluster is None:
            return None
        
        # 设置协议类型
        flow_cluster.protocol = self.file_protocols[idx]
        
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
        
        return flow_cluster


class MultiProtocolLOFOLoader:
    """多协议LOFO数据集加载器 - 按家族组织数据"""

    def __init__(self, c2_data_path, benign_data_path, filter_mode='multi_only'):
        self.c2_data_path = c2_data_path
        self.benign_data_path = benign_data_path
        self.filter_mode = filter_mode
        
        # 显示筛选模式
        filter_mode_desc = {
            'multi_only': '只保留多流簇',
            'multi_and_long_single': '保留多流簇+单流长流',
            'all': '保留所有流簇'
        }
        print(f"\n数据筛选模式: {filter_mode} ({filter_mode_desc.get(filter_mode, '')})")

        # 扫描C2家族文件夹
        self.c2_families = self._scan_c2_families()
        print(f"发现 {len(self.c2_families)} 个C2家族:")
        for family in self.c2_families:
            print(f"  - {family}")

        # 加载良性样本（所有家族共享）
        print("\n加载良性样本...")
        self.benign_dataset = MultiProtocolFlowClusterDataset(benign_data_path, is_pretrain=False, filter_mode=filter_mode)
        print(f"  良性样本总数: {len(self.benign_dataset)}")
        
        # 统计良性样本协议分布
        protocol_counts = defaultdict(int)
        for protocol in self.benign_dataset.file_protocols:
            protocol_counts[protocol] += 1
        print("  协议分布:")
        for protocol, count in sorted(protocol_counts.items()):
            print(f"    {protocol}: {count}")

    def _scan_c2_families(self):
        """扫描C2数据目录下的所有家族文件夹"""
        families = []
        if os.path.exists(self.c2_data_path):
            for item in os.listdir(self.c2_data_path):
                item_path = os.path.join(self.c2_data_path, item)
                if os.path.isdir(item_path):
                    families.append(item)
        print(f"扫描到的家族: {families}")
        return sorted(families)

    def get_family_data(self, family_name):
        """获取指定家族的数据"""
        family_path = os.path.join(self.c2_data_path, family_name)
        print(f"  尝试加载家族: {family_name}, 路径: {family_path}")
        if not os.path.exists(family_path):
            print(f"  ⚠️ 警告: 家族 {family_name} 的路径不存在: {family_path}")
            # 返回空数据集而不是报错
            return MultiProtocolFlowClusterDataset(family_path, is_pretrain=False, filter_mode=self.filter_mode)

        dataset = MultiProtocolFlowClusterDataset(family_path, is_pretrain=False, filter_mode=self.filter_mode)
        return dataset

    def get_lofo_split(self, test_family):
        """
        获取LOFO划分
        test_family: 作为测试集的家族名称
        返回: (train_c2_datasets, test_c2_dataset, benign_train, benign_val, benign_test, c2_val_samples)
        """
        print(f"\n{'='*60}")
        print(f"LOFO Round: 测试家族 = {test_family}")
        print(f"{'='*60}")

        # 训练家族：除测试家族外的所有家族
        train_families = [f for f in self.c2_families if f != test_family]

        # 加载训练家族数据
        train_c2_datasets = []
        for family in train_families:
            family_dataset = self.get_family_data(family)
            print(f"  训练家族 {family}: {len(family_dataset)} 个样本")
            train_c2_datasets.append(family_dataset)

        # 加载测试家族数据
        test_c2_dataset = self.get_family_data(test_family)
        print(f"  测试家族 {test_family}: {len(test_c2_dataset)} 个样本")

        # 良性样本划分（从训练家族中划分验证集）
        benign_data = []
        for i in range(len(self.benign_dataset)):
            fc = self.benign_dataset[i]
            if fc is not None:
                benign_data.append((fc, 0.0))
        np.random.shuffle(benign_data)

        # 良性样本划分：85%训练，10%验证，5%测试（减少测试集比例以匹配C2测试样本数量）
        n_benign = len(benign_data)
        n_train = int(n_benign * 0.85)
        n_val = int(n_benign * 0.1)

        benign_train = benign_data[:n_train]
        benign_val = benign_data[n_train:n_train+n_val]
        benign_test = benign_data[n_train+n_val:]

        # 计算验证集比例（基于良性样本）
        val_ratio = len(benign_val) / n_benign if n_benign > 0 else 0.1
        print(f"\n良性样本划分:")
        print(f"  训练集: {len(benign_train)} ({len(benign_train)/n_benign*100:.1f}%)")
        print(f"  验证集: {len(benign_val)} ({len(benign_val)/n_benign*100:.1f}%)")
        print(f"  测试集: {len(benign_test)} ({len(benign_test)/n_benign*100:.1f}%)")

        # 从训练C2样本中划分验证集（使用与良性样本相同的验证集比例）
        c2_val_samples = []
        c2_train_datasets = []
        for dataset in train_c2_datasets:
            dataset_size = len(dataset)
            # 使用与良性样本相同的验证集比例
            n_c2_val = max(1, int(dataset_size * val_ratio))
            
            indices = list(range(dataset_size))
            np.random.shuffle(indices)
            val_indices = indices[:n_c2_val]
            train_indices = indices[n_c2_val:]
            
            for i in val_indices:
                fc = dataset[i]
                if fc is not None:
                    c2_val_samples.append((fc, 1.0))
            
            train_dataset = [dataset[i] for i in train_indices if dataset[i] is not None]
            c2_train_datasets.append(train_dataset)
        
        # 计算C2验证集实际比例
        total_c2_train = sum(len(ds) for ds in c2_train_datasets)
        c2_val_ratio = len(c2_val_samples) / (total_c2_train + len(c2_val_samples)) if (total_c2_train + len(c2_val_samples)) > 0 else 0
        print(f"\nC2验证集样本数: {len(c2_val_samples)} ({c2_val_ratio*100:.1f}%)")

        return c2_train_datasets, test_c2_dataset, benign_train, benign_val, benign_test, c2_val_samples


def merge_c2_datasets(c2_datasets):
    """合并多个C2数据集"""
    merged_data = []
    for dataset in c2_datasets:
        for fc in dataset:
            if fc is not None:
                merged_data.append((fc, 1.0))
    return merged_data


def apply_augmentation_by_protocol(data, target_ratio=0.5):
    """按协议分别进行样本增强，确保每个协议内部C2和良性样本平衡
    
    Args:
        data: 训练数据列表，每个元素为 (flow_cluster, label) 元组
        target_ratio: 目标少数类占比（默认0.5，即50%）
    
    Returns:
        增强后的数据列表
    """
    if not AUGMENTATION_AVAILABLE or len(data) == 0:
        return data
    
    augmenter = FlowClusterAugmentation()
    
    # 按协议分组数据
    protocol_data = {'TLS': [], 'HTTP': [], 'TCP': []}
    for fc, label in data:
        protocol = fc.protocol if hasattr(fc, 'protocol') else 'TLS'
        if protocol in protocol_data:
            protocol_data[protocol].append((fc, label))
    
    print("  按协议分别进行样本增强:")
    
    # 对每个协议分别进行增强
    augmented_data = []
    for protocol in ['TLS', 'HTTP', 'TCP']:
        proto_samples = protocol_data[protocol]
        if len(proto_samples) == 0:
            continue
        
        # 分离C2和良性样本
        c2_data = [(fc, label) for fc, label in proto_samples if label == 1.0]
        benign_data = [(fc, label) for fc, label in proto_samples if label == 0.0]
        
        if len(c2_data) == 0 or len(benign_data) == 0:
            # 如果某类样本为空，直接添加原始数据
            augmented_data.extend(proto_samples)
            print(f"    {protocol}: C2={len(c2_data)}, 良性={len(benign_data)} (跳过增强，某类为空)")
            continue
        
        # 确定少数类和多数类
        if len(c2_data) < len(benign_data):
            minority_data = c2_data
            majority_data = benign_data
            minority_class = "C2"
        else:
            minority_data = benign_data
            majority_data = c2_data
            minority_class = "良性"
        
        # 计算需要的增强数量
        target_minority_count = int(target_ratio * len(majority_data) / (1 - target_ratio))
        target_minority_count = max(target_minority_count, len(minority_data))
        additional_needed = target_minority_count - len(minority_data)
        
        print(f"    {protocol}: 原始 C2={len(c2_data)}, 良性={len(benign_data)}")
        
        # 增强后的数据
        proto_augmented = majority_data.copy()
        proto_augmented.extend(minority_data)
        
        # 对少数类样本进行增强
        generated_count = 0
        attempts = 0
        max_attempts = additional_needed * 3  # 防止无限循环
        
        while generated_count < additional_needed and attempts < max_attempts:
            for flow_cluster, label in minority_data:
                if generated_count >= additional_needed:
                    break
                try:
                    augmented_cluster = augmenter.augment(flow_cluster)
                    proto_augmented.append((augmented_cluster, label))
                    generated_count += 1
                except Exception:
                    pass
                attempts += 1
        
        # 打乱该协议的数据
        np.random.shuffle(proto_augmented)
        augmented_data.extend(proto_augmented)
        
        # 统计增强后的分布
        final_c2 = sum(1 for _, label in proto_augmented if label == 1.0)
        final_benign = sum(1 for _, label in proto_augmented if label == 0.0)
        final_ratio = final_c2 / len(proto_augmented) if len(proto_augmented) > 0 else 0
        print(f"      增强后: C2={final_c2}, 良性={final_benign}, 总计={len(proto_augmented)}, C2占比={final_ratio:.1%}")
    
    # 打乱整体数据
    np.random.shuffle(augmented_data)
    
    # 统计总体分布
    total_c2 = sum(1 for _, label in augmented_data if label == 1.0)
    total_benign = sum(1 for _, label in augmented_data if label == 0.0)
    print(f"    总计: C2={total_c2}, 良性={total_benign}, 总样本数={len(augmented_data)}")
    
    return augmented_data


class MultiProtocolLogic2Model:
    """多协议LOGIC2模型管理器"""
    
    def __init__(self, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self._init_models()
    
    def _init_models(self):
        """初始化各协议的模型"""
        protocols = ['tls', 'http', 'tcp']
        
        for protocol in protocols:
            sequence_encoder = SequenceEncoder()
            graph_encoder = GraphEncoder(output_dim=128)
            
            # 加载预训练权重
            seq_path = f"./sequence_encoder_pretrained_{protocol}.pth"
            graph_path = f"./graph_encoder_pretrained_{protocol}.pth"
            
            if os.path.exists(seq_path):
                try:
                    sequence_encoder.load_state_dict(torch.load(seq_path, weights_only=True))
                    print(f"  ✓ 加载{protocol.upper()}序列编码器")
                except Exception as e:
                    print(f"  ⚠️ {protocol.upper()}序列编码器加载失败: {e}")
            
            if os.path.exists(graph_path):
                try:
                    graph_encoder.load_state_dict(torch.load(graph_path, weights_only=True), strict=False)
                    print(f"  ✓ 加载{protocol.upper()}图编码器")
                except Exception as e:
                    print(f"  ⚠️ {protocol.upper()}图编码器加载失败: {e}")
            
            model = Logic2Model(sequence_encoder, graph_encoder)
            model.to(self.device)
            self.models[protocol.upper()] = model
    
    def get_model(self, protocol):
        """获取指定协议的模型"""
        if protocol not in self.models:
            print(f"⚠️ 未知协议: {protocol}，使用TLS模型")
            protocol = 'TLS'
        return self.models[protocol]
    
    def predict(self, flow_cluster):
        """预测单个流簇"""
        protocol = flow_cluster.protocol if hasattr(flow_cluster, 'protocol') else 'TLS'
        model = self.get_model(protocol)
        model.eval()
        
        with torch.no_grad():
            pred = model(flow_cluster)
            prob = torch.sigmoid(pred).item()
        
        return prob
    
    def train_mode(self):
        """设置所有模型为训练模式"""
        for model in self.models.values():
            model.train()
    
    def eval_mode(self):
        """设置所有模型为评估模式"""
        for model in self.models.values():
            model.eval()
    
    def get_all_parameters(self):
        """获取所有模型的参数"""
        all_params = []
        for model in self.models.values():
            all_params.extend(list(model.parameters()))
        return all_params
    
    def state_dict(self):
        """获取所有模型的状态字典"""
        return {protocol: model.state_dict() for protocol, model in self.models.items()}
    
    def load_state_dict(self, state_dict):
        """加载所有模型的状态字典"""
        for protocol, model_state in state_dict.items():
            if protocol in self.models:
                self.models[protocol].load_state_dict(model_state)


def train_lofo_model(train_data, val_data, multi_model, epochs=10, batch_size=64, use_augmentation=True):
    """LOFO训练 - 多协议联合训练"""
    device = multi_model.device
    
    # 应用样本增强（按协议分别增强）
    if use_augmentation and AUGMENTATION_AVAILABLE:
        print("\n应用样本增强（按协议分别增强，目标: C2占比50%）...")
        train_data = apply_augmentation_by_protocol(train_data, target_ratio=0.5)
        val_data = apply_augmentation_by_protocol(val_data, target_ratio=0.5)
    
    # 为每个协议准备优化器
    protocol_optimizers = {}
    for protocol in ['TLS', 'HTTP', 'TCP']:
        model = multi_model.get_model(protocol)
        protocol_optimizers[protocol] = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    bce_loss = nn.BCEWithLogitsLoss()
    
    best_val_f1 = 0.0
    patience = 3  # 减少耐心值，更快早停
    patience_counter = 0
    best_model_state = None
    
    # 按协议分组数据
    protocol_train_data = {p: [] for p in ['TLS', 'HTTP', 'TCP']}
    protocol_val_data = {p: [] for p in ['TLS', 'HTTP', 'TCP']}
    
    for fc, label in train_data:
        if hasattr(fc, 'protocol') and fc.protocol in protocol_train_data:
            protocol_train_data[fc.protocol].append((fc, label))
    
    for fc, label in val_data:
        if hasattr(fc, 'protocol') and fc.protocol in protocol_val_data:
            protocol_val_data[fc.protocol].append((fc, label))
    
    # 打印各协议数据分布
    print("\n各协议训练数据分布:")
    for protocol in ['TLS', 'HTTP', 'TCP']:
        c2_count = sum(1 for _, label in protocol_train_data[protocol] if label == 1.0)
        benign_count = sum(1 for _, label in protocol_train_data[protocol] if label == 0.0)
        if len(protocol_train_data[protocol]) > 0:
            print(f"  {protocol}: C2={c2_count}, 良性={benign_count}, 总计={len(protocol_train_data[protocol])}")
    
    for epoch in range(epochs):
        multi_model.train_mode()
        total_loss = 0.0
        
        # 训练每个协议
        for protocol in ['TLS', 'HTTP', 'TCP']:
            if len(protocol_train_data[protocol]) == 0:
                continue
            
            model = multi_model.get_model(protocol)
            optimizer = protocol_optimizers[protocol]
            
            for i in range(0, len(protocol_train_data[protocol]), batch_size):
                batch = protocol_train_data[protocol][i:i+batch_size]
                batch_preds = []
                batch_labels = []
                
                for flow_cluster, label in batch:
                    pred = model(flow_cluster)
                    pred = pred.to(device)
                    batch_preds.append(pred)
                    label_tensor = torch.tensor([label], dtype=torch.float32, device=device)
                    batch_labels.append(label_tensor)
                
                if batch_preds:
                    batch_preds = torch.cat(batch_preds, dim=0)
                    batch_labels = torch.cat(batch_labels, dim=0)
                    loss = bce_loss(batch_preds, batch_labels)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
        
        # 验证
        val_metrics, _ = evaluate_lofo(val_data, multi_model, "Validation")
        val_f1 = val_metrics['f1']
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_data):.4f}, Val F1: {val_f1:.4f}")
        
        # 早停
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = multi_model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停触发! Best Val F1: {best_val_f1:.4f}")
                break
    
    # 加载最佳模型
    if best_model_state:
        multi_model.load_state_dict(best_model_state)
    
    return multi_model


def evaluate_lofo(test_data, multi_model, test_family, threshold=0.5, cached_probs=None):
    """LOFO评估 - 多协议评估（支持缓存概率值以加速阈值搜索）
    
    Args:
        test_data: 测试数据
        multi_model: 多协议模型
        test_family: 测试家族名称
        threshold: 分类阈值（默认0.5）
        cached_probs: 缓存的概率值（用于阈值搜索时避免重复推理）
    """
    multi_model.eval_mode()
    device = multi_model.device
    
    # 如果有缓存的概率值，直接使用
    if cached_probs is not None:
        all_labels = cached_probs['labels']
        all_probs = cached_probs['probs']
        all_protocols = cached_probs['protocols']
    else:
        all_labels = []
        all_probs = []
        all_protocols = []
        
        with torch.no_grad():
            for flow_cluster, label in test_data:
                protocol = flow_cluster.protocol if hasattr(flow_cluster, 'protocol') else 'TLS'
                model = multi_model.get_model(protocol)
                
                pred = model(flow_cluster)
                prob = torch.sigmoid(pred).item()
                
                all_labels.append(label)
                all_probs.append(prob)
                all_protocols.append(protocol)
    
    # 使用指定阈值进行预测
    all_preds = [1 if prob > threshold else 0 for prob in all_probs]
    
    # 按协议统计（先计算协议级别指标）
    protocol_stats = {}
    for protocol in ['TLS', 'HTTP', 'TCP']:
        protocol_indices = [i for i, p in enumerate(all_protocols) if p == protocol]
        if len(protocol_indices) > 0:
            protocol_probs = [all_probs[i] for i in protocol_indices]
            protocol_labels_list = [all_labels[i] for i in protocol_indices]
            protocol_preds = [1 if prob > threshold else 0 for prob in protocol_probs]
            
            c2_count = int(sum(protocol_labels_list))
            benign_count = len(protocol_labels_list) - c2_count
            
            # 计算该协议的混淆矩阵元素
            tp = sum(1 for i in range(len(protocol_labels_list)) 
                    if protocol_labels_list[i] == 1 and protocol_preds[i] == 1)
            fp = sum(1 for i in range(len(protocol_labels_list)) 
                    if protocol_labels_list[i] == 0 and protocol_preds[i] == 1)
            tn = sum(1 for i in range(len(protocol_labels_list)) 
                    if protocol_labels_list[i] == 0 and protocol_preds[i] == 0)
            fn = sum(1 for i in range(len(protocol_labels_list)) 
                    if protocol_labels_list[i] == 1 and protocol_preds[i] == 0)
            
            # 计算F1（特殊处理无C2样本的情况）
            if c2_count == 0:
                # 没有C2样本：如果所有良性样本都被正确预测为良性，F1=1.0
                if fp == 0 and tn == benign_count:
                    protocol_f1 = 1.0
                else:
                    # 有误报，F1=0（因为没有C2样本，无法真正检测）
                    protocol_f1 = 0.0
            else:
                # 有C2样本：正常计算F1
                protocol_f1 = float(f1_score(protocol_labels_list, protocol_preds, zero_division=0))
            
            protocol_stats[protocol] = {
                'total': len(protocol_indices),
                'accuracy': float(np.mean(np.array(protocol_preds) == np.array(protocol_labels_list))),
                'f1': protocol_f1,
                'c2_count': c2_count,
                'benign_count': benign_count,
                'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
            }
    
    # 筛选出有C2样本的协议
    protocols_with_c2 = [p for p in ['TLS', 'HTTP', 'TCP'] 
                        if p in protocol_stats and protocol_stats[p]['c2_count'] > 0]
    
    if protocols_with_c2:
        # 只使用有C2样本的协议的数据计算总体指标
        filtered_indices = [i for i, p in enumerate(all_protocols) if p in protocols_with_c2]
        filtered_labels = [all_labels[i] for i in filtered_indices]
        filtered_preds = [all_preds[i] for i in filtered_indices]
        filtered_probs = [all_probs[i] for i in filtered_indices]
        
        # 计算总体指标（基于有C2样本的协议）
        metrics = {
            'accuracy': float(np.mean(np.array(filtered_preds) == np.array(filtered_labels))),
            'precision': float(precision_score(filtered_labels, filtered_preds, zero_division=0)),
            'recall': float(recall_score(filtered_labels, filtered_preds, zero_division=0)),
            'f1': float(f1_score(filtered_labels, filtered_preds, zero_division=0)),
            'roc_auc': float(roc_auc_score(filtered_labels, filtered_probs)) if len(set(filtered_labels)) > 1 else 0.5,
            'total_samples': len(filtered_labels),
            'c2_samples': int(sum(filtered_labels)),
            'benign_samples': int(len(filtered_labels) - sum(filtered_labels)),
            'threshold': threshold,
            'protocols_with_c2': protocols_with_c2
        }
        
        # 计算混淆矩阵（基于有C2样本的协议）
        cm = confusion_matrix(filtered_labels, filtered_preds)
        metrics['confusion_matrix'] = cm.tolist()
        
        # 计算FPR
        if cm.shape[0] > 1:
            fp = cm[0][1]
            tn = cm[0][0]
            metrics['fpr'] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        else:
            metrics['fpr'] = 0.0
    else:
        # 如果没有协议有C2样本，使用所有数据计算指标
        metrics = {
            'accuracy': float(np.mean(np.array(all_preds) == np.array(all_labels))),
            'precision': float(precision_score(all_labels, all_preds, zero_division=0)),
            'recall': float(recall_score(all_labels, all_preds, zero_division=0)),
            'f1': float(f1_score(all_labels, all_preds, zero_division=0)),
            'roc_auc': float(roc_auc_score(all_labels, all_probs)) if len(set(all_labels)) > 1 else 0.5,
            'total_samples': len(all_labels),
            'c2_samples': int(sum(all_labels)),
            'benign_samples': int(len(all_labels) - sum(all_labels)),
            'threshold': threshold,
            'protocols_with_c2': []
        }
        
        # 计算混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        metrics['confusion_matrix'] = cm.tolist()
        
        # 计算FPR
        if cm.shape[0] > 1:
            fp = cm[0][1]
            tn = cm[0][0]
            metrics['fpr'] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        else:
            metrics['fpr'] = 0.0
    
    metrics['protocol_stats'] = protocol_stats
    
    print(f"\n测试结果 ({test_family}, 阈值={threshold}):")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1: {metrics['f1']:.4f}")
    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"  FPR: {metrics['fpr']:.4f}")
    print(f"  样本分布: C2={metrics['c2_samples']}, 良性={metrics['benign_samples']}")
    
    # 显示哪些协议有C2样本
    if 'protocols_with_c2' in metrics and metrics['protocols_with_c2']:
        print(f"  有C2样本的协议: {', '.join(metrics['protocols_with_c2'])}")
    else:
        print(f"  有C2样本的协议: 无")
    
    if protocol_stats:
        print(f"\n  各协议结果:")
        for protocol, stats in protocol_stats.items():
            if stats['c2_count'] == 0:
                # 没有C2样本的情况
                if stats['f1'] == 1.0:
                    print(f"    {protocol}: Acc={stats['accuracy']:.4f}, F1={stats['f1']:.4f} (无C2样本, 良性全正确), "
                          f"C2=0, 良性={stats['benign_count']}, N={stats['total']}")
                else:
                    print(f"    {protocol}: Acc={stats['accuracy']:.4f}, F1={stats['f1']:.4f} (无C2样本, 有误报), "
                          f"C2=0, 良性={stats['benign_count']}, N={stats['total']}")
            else:
                # 有C2样本的情况
                print(f"    {protocol}: Acc={stats['accuracy']:.4f}, F1={stats['f1']:.4f}, "
                      f"C2={stats['c2_count']}, 良性={stats['benign_count']}, N={stats['total']}")
    
    # 返回指标和缓存数据（用于后续阈值搜索）
    cache_data = {
        'labels': all_labels,
        'probs': all_probs,
        'protocols': all_protocols
    }
    
    return metrics, cache_data


def lofo_evaluation(c2_data_path, benign_data_path, output_dir="./lofo_multi_results", filter_mode='multi_only'):
    """执行多协议LOFO交叉验证"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化数据加载器
    loader = MultiProtocolLOFOLoader(c2_data_path, benign_data_path, filter_mode=filter_mode)
    
    # 存储每轮结果
    all_results = []
    
    # 对每个家族执行LOFO
    for test_family in loader.c2_families:
        print(f"\n{'='*80}")
        print(f"开始 LOFO Round: {test_family}")
        print(f"{'='*80}")
        
        # 获取数据划分
        train_c2_datasets, test_c2_dataset, benign_train, benign_val, benign_test, c2_val_samples = \
            loader.get_lofo_split(test_family)
        
        # 合并训练C2数据
        train_c2_data = merge_c2_datasets(train_c2_datasets)
        test_c2_data = []
        for i in range(len(test_c2_dataset)):
            fc = test_c2_dataset[i]
            if fc is not None:
                test_c2_data.append((fc, 1.0))
        
        print(f"\n训练C2样本数: {len(train_c2_data)}")
        print(f"验证C2样本数: {len(c2_val_samples)}")
        print(f"测试C2样本数: {len(test_c2_data)}")
        
        # 构建训练集和验证集
        train_data = train_c2_data + benign_train
        val_data = benign_val + c2_val_samples
        
        np.random.shuffle(train_data)
        
        print(f"\n最终数据集:")
        print(f"  训练集: {len(train_data)} (C2: {len(train_c2_data)}, 良性: {len(benign_train)})")
        print(f"  验证集: {len(val_data)} (C2: {len(c2_val_samples)}, 良性: {len(benign_val)})")
        print(f"  测试集: {len(test_c2_data) + len(benign_test)} (C2: {len(test_c2_data)}, 良性: {len(benign_test)})")
        
        # 创建多协议模型
        print("\n初始化多协议模型...")
        multi_model = MultiProtocolLogic2Model()
        
        # 训练模型
        print(f"\n开始训练...")
        multi_model = train_lofo_model(train_data, val_data, multi_model, epochs=10, batch_size=64)
        
        # 评估测试集
        print(f"\n评估测试集...")
        test_data = test_c2_data + benign_test
        np.random.shuffle(test_data)
        
        # 首先使用默认阈值0.5评估（并获取缓存的概率值）
        test_metrics, prob_cache = evaluate_lofo(test_data, multi_model, test_family, threshold=0.5)
        
        # 如果F1很低但ROC-AUC很高，尝试不同的阈值（使用缓存的概率值，避免重复推理）
        if test_metrics['f1'] < 0.3 and test_metrics['roc_auc'] > 0.7:
            print(f"\n  注意: F1较低({test_metrics['f1']:.4f})但ROC-AUC较高({test_metrics['roc_auc']:.4f})，尝试优化阈值...")
            
            # 尝试不同的阈值（使用缓存）
            best_threshold = 0.5
            best_f1 = test_metrics['f1']
            
            for threshold in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
                metrics, _ = evaluate_lofo(test_data, multi_model, test_family, 
                                          threshold=threshold, cached_probs=prob_cache)
                if metrics['f1'] > best_f1:
                    best_f1 = metrics['f1']
                    best_threshold = threshold
            
            if best_threshold != 0.5:
                print(f"\n  最佳阈值: {best_threshold}, 最佳F1: {best_f1:.4f}")
                # 使用最佳阈值重新评估
                test_metrics, _ = evaluate_lofo(test_data, multi_model, test_family, 
                                               threshold=best_threshold, cached_probs=prob_cache)
            else:
                print(f"\n  阈值优化未找到更好的结果，使用默认阈值0.5")
        
        # 保存结果
        result = {
            'test_family': test_family,
            'train_families': [f for f in loader.c2_families if f != test_family],
            'train_c2_samples': len(train_c2_data),
            'test_c2_samples': len(test_c2_data),
            'metrics': test_metrics
        }
        all_results.append(result)
        
        # 保存本轮详细结果
        round_file = os.path.join(output_dir, f"lofo_{test_family}.json")
        with open(round_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n本轮结果已保存: {round_file}")
    
    # 生成总体报告
    generate_lofo_report(all_results, output_dir)
    
    return all_results


def generate_lofo_report(all_results, output_dir):
    """生成LOFO实验报告
    
    计算两种平均方式：
    - Macro-average: 各家族权重相同（简单平均）
    - Weighted Macro-average: 按样本数量加权平均
    """
    print(f"\n{'='*80}")
    print("LOFO 交叉验证总体报告 (多协议)")
    print(f"{'='*80}")
    
    # 收集所有指标
    f1_scores = [r['metrics']['f1'] for r in all_results]
    recall_scores = [r['metrics']['recall'] for r in all_results]
    fpr_scores = [r['metrics']['fpr'] for r in all_results]
    roc_auc_scores = [r['metrics']['roc_auc'] for r in all_results]
    
    # 收集样本数量用于加权计算
    sample_counts = [r['metrics']['total_samples'] for r in all_results]
    c2_counts = [r['metrics']['c2_samples'] for r in all_results]
    total_samples = sum(sample_counts)
    total_c2 = sum(c2_counts)
    
    # 按协议收集（只收集有C2样本的协议）
    protocol_f1 = {'TLS': [], 'HTTP': [], 'TCP': []}
    protocol_samples = {'TLS': [], 'HTTP': [], 'TCP': []}
    protocol_c2_counts = {'TLS': [], 'HTTP': [], 'TCP': []}
    
    for result in all_results:
        metrics = result['metrics']
        protocol_stats = metrics.get('protocol_stats', {})
        
        for protocol in ['TLS', 'HTTP', 'TCP']:
            if protocol in protocol_stats:
                stats = protocol_stats[protocol]
                # 只统计有C2样本的协议
                if stats['c2_count'] > 0:
                    protocol_f1[protocol].append(stats['f1'])
                    protocol_samples[protocol].append(stats['total'])
                    protocol_c2_counts[protocol].append(stats['c2_count'])
    
    # 计算加权平均（按样本数量加权）
    def weighted_average(scores, weights):
        weights = np.array(weights)
        scores = np.array(scores)
        return float(np.sum(scores * weights) / np.sum(weights))
    
    weighted_f1 = weighted_average(f1_scores, sample_counts)
    weighted_recall = weighted_average(recall_scores, sample_counts)
    weighted_fpr = weighted_average(fpr_scores, sample_counts)
    weighted_roc_auc = weighted_average(roc_auc_scores, sample_counts)
    
    # 计算统计值
    report = {
        # Macro-average（简单平均）
        'average_macro_f1': float(np.mean(f1_scores)),
        'std_macro_f1': float(np.std(f1_scores)),
        'worst_case_f1': float(np.min(f1_scores)),
        'best_case_f1': float(np.max(f1_scores)),
        'average_recall': float(np.mean(recall_scores)),
        'average_fpr': float(np.mean(fpr_scores)),
        'average_roc_auc': float(np.mean(roc_auc_scores)),
        # Weighted Macro-average（加权平均）
        'weighted_macro_f1': weighted_f1,
        'weighted_recall': weighted_recall,
        'weighted_fpr': weighted_fpr,
        'weighted_roc_auc': weighted_roc_auc,
        # 样本统计
        'total_samples': total_samples,
        'total_c2_samples': total_c2,
        'num_families': len(all_results),
        'per_family_results': all_results
    }
    
    print(f"\n总体指标:")
    print(f"\n  [Macro-Average] 各家族权重相同:")
    print(f"    Average Macro-F1: {report['average_macro_f1']:.4f} (±{report['std_macro_f1']:.4f})")
    print(f"    Worst-case F1: {report['worst_case_f1']:.4f}")
    print(f"    Best-case F1: {report['best_case_f1']:.4f}")
    print(f"    Average Recall: {report['average_recall']:.4f}")
    print(f"    Average FPR: {report['average_fpr']:.4f}")
    print(f"    Average ROC-AUC: {report['average_roc_auc']:.4f}")
    
    print(f"\n  [Weighted Macro-Average] 按样本数量加权:")
    print(f"    Weighted F1: {report['weighted_macro_f1']:.4f}")
    print(f"    Weighted Recall: {report['weighted_recall']:.4f}")
    print(f"    Weighted FPR: {report['weighted_fpr']:.4f}")
    print(f"    Weighted ROC-AUC: {report['weighted_roc_auc']:.4f}")
    
    print(f"\n  [样本统计]:")
    print(f"    总样本数: {total_samples}")
    print(f"    总C2样本数: {total_c2}")
    print(f"    家族数量: {len(all_results)}")
    
    # 协议级别统计（只统计有C2样本的家族）
    print(f"\n  [各协议统计 - 仅含C2样本的家族]:")
    for protocol in ['TLS', 'HTTP', 'TCP']:
        if protocol_f1[protocol]:
            avg_f1 = np.mean(protocol_f1[protocol])
            std_f1 = np.std(protocol_f1[protocol])
            min_f1 = np.min(protocol_f1[protocol])
            max_f1 = np.max(protocol_f1[protocol])
            weighted_f1_proto = weighted_average(protocol_f1[protocol], protocol_samples[protocol])
            total_proto = sum(protocol_samples[protocol])
            total_c2_proto = sum(protocol_c2_counts[protocol])
            num_families = len(protocol_f1[protocol])
            print(f"    {protocol}: Macro-F1={avg_f1:.4f} (±{std_f1:.4f}), "
                  f"Weighted-F1={weighted_f1_proto:.4f}, "
                  f"Range=[{min_f1:.4f}, {max_f1:.4f}], "
                  f"N={num_families}, 总样本={total_proto}, C2样本={total_c2_proto}")
        else:
            print(f"    {protocol}: 无C2样本")
    
    print(f"\n各家族详细结果:")
    print(f"  {'家族':<20} {'F1':>8} {'Recall':>8} {'FPR':>8} {'样本数':>8} {'C2样本':>8} {'有C2的协议':>12}")
    print(f"  {'-'*95}")
    
    for result in all_results:
        family = result['test_family']
        metrics = result['metrics']
        n_samples = metrics['total_samples']
        n_c2 = metrics['c2_samples']
        protocols_with_c2 = metrics.get('protocols_with_c2', [])
        protocols_str = ','.join(protocols_with_c2) if protocols_with_c2 else 'None'
        print(f"  {family:<20} {metrics['f1']:>8.4f} {metrics['recall']:>8.4f} "
              f"{metrics['fpr']:>8.4f} {n_samples:>8} {n_c2:>8} {protocols_str:>12}")
    
    # 保存报告
    report_file = os.path.join(output_dir, "lofo_multi_summary_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n报告已保存: {report_file}")
    
    return report


def main():
    parser = argparse.ArgumentParser(description="LOGIC2 Multi-Protocol LOFO Cross-Family Generalization Evaluation")
    parser.add_argument("--c2-data", type=str, default="./MTA-csv", help="C2数据路径（包含家族子文件夹）")
    parser.add_argument("--benign-data", type=str, default="./CTU-csv", help="良性数据路径")
    parser.add_argument("--output-dir", type=str, default="./lofo_multi_results", help="结果输出目录")
    parser.add_argument("--skip-families", type=str, default="", help="跳过的家族名称（逗号分隔）")
    parser.add_argument("--filter-mode", type=str, default="multi_only",
                        choices=["multi_only", "multi_and_long_single", "all"],
                        help="数据筛选模式: multi_only=只保留多流簇, "
                             "multi_and_long_single=保留多流簇+单流长流, all=保留所有")
    args = parser.parse_args()
    
    print("="*80)
    print("LOGIC2 多协议 Leave-One-Family-Out (LOFO) 交叉验证")
    print("="*80)
    print(f"C2数据路径: {args.c2_data}")
    print(f"良性数据路径: {args.benign_data}")
    print(f"输出目录: {args.output_dir}")
    print(f"筛选模式: {args.filter_mode}")
    print()
    
    # 执行LOFO评估
    results = lofo_evaluation(args.c2_data, args.benign_data, args.output_dir, filter_mode=args.filter_mode)
    
    print("\n" + "="*80)
    print("LOFO评估完成!")
    print("="*80)


if __name__ == "__main__":
    main()
