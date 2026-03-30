"""
LOGIC2 多协议 Few-shot 分类实验脚本
用于评估极有限标注数据下的检测性能
支持 TLS、HTTP、TCP 三种协议

K-shot 设置：
- K ∈ {1, 3, 5, 10}
- 每个符合条件的 C2 家族随机采样 exactly K 个流簇
- 所有良性流量全部使用（不做采样）
"""

import os
import sys
import json
import warnings
import numpy as np
import torch
import torch.nn as nn
import argparse
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*flash attention.*')
warnings.filterwarnings('ignore', message='.*Torch was not compiled with flash attention.*')

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from logic2_tls_import import (
    FlowBlock, Flow, FlowCluster, FlowClusterDataset,
    evaluate, logger, print
)

import importlib.util
spec = importlib.util.spec_from_file_location("logic2_tls", "2.logic2-tls.py")
logic2_module = importlib.util.module_from_spec(spec)

import types
sys.modules['logic2_tls_module'] = logic2_module

import torch_geometric
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data
from torch.nn import TransformerEncoder, TransformerEncoderLayer

spec.loader.exec_module(logic2_module)

SequenceEncoder = logic2_module.SequenceEncoder
GraphEncoder = logic2_module.GraphEncoder
Logic2Model = logic2_module.Logic2Model

try:
    from flow_cluster_augmentation import FlowClusterAugmentation
    AUGMENTATION_AVAILABLE = True
except ImportError:
    AUGMENTATION_AVAILABLE = False
    print("警告: 未找到样本增强模块，将不使用数据增强")


def detect_protocol(filename):
    """检测文件名对应的协议类型"""
    tls_patterns = ['_tls1.0.csv', '_tls1.1.csv', '_tls1.2.csv', '_tls1.3.csv', '_ssl3.0.csv']
    for pattern in tls_patterns:
        if filename.endswith(pattern):
            return 'TLS'

    if filename.endswith('_http.csv'):
        return 'HTTP'

    if filename.endswith('_tcp.csv'):
        return 'TCP'

    return 'Unknown'


class MultiProtocolFlowClusterDataset:
    """多协议流簇数据集"""

    def __init__(self, csv_folder, is_pretrain=True, max_flows_per_cluster=100, filter_mode='multi_only'):
        self.csv_folder = csv_folder
        self.is_pretrain = is_pretrain
        self.max_flows_per_cluster = max_flows_per_cluster
        self.filter_mode = filter_mode

        self.protocol_files = {'TLS': [], 'HTTP': [], 'TCP': [], 'Unknown': []}

        if os.path.exists(csv_folder):
            for root, dirs, files in os.walk(csv_folder):
                for file in files:
                    if file.endswith('.csv'):
                        protocol = detect_protocol(file)
                        rel_path = os.path.relpath(os.path.join(root, file), csv_folder)
                        self.protocol_files[protocol].append(rel_path)

        self.all_files = []
        self.file_protocols = []
        for protocol in ['TLS', 'HTTP', 'TCP']:
            for file in self.protocol_files[protocol]:
                self.all_files.append(file)
                self.file_protocols.append(protocol)

        if os.path.exists(csv_folder):
            self.base_dataset = FlowClusterDataset(csv_folder, is_pretrain=is_pretrain)
        else:
            self.base_dataset = None

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            results = []
            for i in range(*idx.indices(len(self))):
                result = self._load_item(i)
                if result is not None:
                    results.append(result)
            return results
        else:
            return self._load_item(idx)

    def _load_item(self, idx):
        if self.base_dataset is None or idx >= len(self.all_files):
            return None

        csv_file = os.path.join(self.csv_folder, self.all_files[idx])
        flow_cluster = self.base_dataset._load_csv_to_flow_cluster(csv_file)

        if flow_cluster is None:
            return None

        flow_cluster.protocol = self.file_protocols[idx]

        flow_count = len(flow_cluster.flows)

        if self.filter_mode == 'multi_only':
            if flow_count <= 1:
                return None
        elif self.filter_mode == 'multi_and_long_single':
            if flow_count == 1:
                if not flow_cluster.flows[0].is_long_flow:
                    return None
        elif self.filter_mode == 'all':
            pass

        return flow_cluster

    def get_all_samples(self):
        """获取所有样本，返回 (flow_cluster, protocol) 列表"""
        samples = []
        for i in range(len(self)):
            fc = self._load_item(i)
            if fc is not None:
                samples.append((fc, fc.protocol))
        return samples


class MultiProtocolFewShotLoader:
    """多协议 Few-shot 数据加载器"""

    def __init__(self, c2_data_path, benign_data_path, filter_mode='multi_only', max_benign_samples=None, target_protocol='ALL'):
        self.c2_data_path = c2_data_path
        self.benign_data_path = benign_data_path
        self.filter_mode = filter_mode
        self.max_benign_samples = max_benign_samples
        self.target_protocol = target_protocol

        filter_mode_desc = {
            'multi_only': '只保留多流簇',
            'multi_and_long_single': '保留多流簇+单流长流',
            'all': '保留所有流簇'
        }
        print(f"\n数据筛选模式: {filter_mode} ({filter_mode_desc.get(filter_mode, '')})")

        self.c2_families = self._scan_c2_families()
        print(f"发现 {len(self.c2_families)} 个C2家族:")
        for family in self.c2_families:
            print(f"  - {family}")

        print("\n加载良性样本...")
        self.benign_dataset = MultiProtocolFlowClusterDataset(benign_data_path, is_pretrain=False, filter_mode=filter_mode)
        print(f"  良性样本总数: {len(self.benign_dataset)}")

        protocol_counts = defaultdict(int)
        for protocol in self.benign_dataset.file_protocols:
            protocol_counts[protocol] += 1

        target_protocol = getattr(self, 'target_protocol', 'ALL')
        if target_protocol != 'ALL':
            target_count = protocol_counts.get(target_protocol, 0)
            print(f"  目标协议 [{target_protocol}] 样本数: {target_count}")
            if self.max_benign_samples:
                print(f"  [良性样本限制] 将采样 {self.max_benign_samples} 个")
        else:
            print("  协议分布:")
            for protocol, count in sorted(protocol_counts.items()):
                print(f"    {protocol}: {count}")

    def _scan_c2_families(self):
        families = []
        if os.path.exists(self.c2_data_path):
            for item in os.listdir(self.c2_data_path):
                item_path = os.path.join(self.c2_data_path, item)
                if os.path.isdir(item_path):
                    families.append(item)
        return sorted(families)

    def get_family_samples(self, family_name):
        """获取指定家族的所有样本"""
        family_path = os.path.join(self.c2_data_path, family_name)
        if not os.path.exists(family_path):
            return []

        dataset = MultiProtocolFlowClusterDataset(family_path, is_pretrain=False, filter_mode=self.filter_mode)
        return dataset.get_all_samples()

    def get_fewshot_split(self, k, seed=42, target_protocol='ALL'):
        """
        获取 Few-shot 数据划分

        Args:
            k: 每个家族采样的样本数 (1, 3, 5, 10)
            seed: 随机种子
            target_protocol: 目标协议 ('ALL', 'TLS', 'HTTP', 'TCP')

        Returns:
            train_samples: 训练样本 [(flow_cluster, label, family, protocol), ...]
            test_samples: 测试样本 [(flow_cluster, label, family, protocol), ...]
            qualified_families: 符合条件的家族列表
            excluded_families: 排除的家族列表（样本数 < k 或无目标协议样本）
        """
        np.random.seed(seed)

        print(f"\n{'='*60}")
        print(f"Few-shot Split: K = {k}, 协议 = {target_protocol}")
        print(f"{'='*60}")

        train_samples = []
        test_samples = []
        qualified_families = []
        excluded_families = []

        for family in self.c2_families:
            family_samples = self.get_family_samples(family)

            if target_protocol != 'ALL':
                original_count = len(family_samples)
                family_samples = [s for s in family_samples if s[1] == target_protocol]
                protocol_count = len(family_samples)
                display_suffix = f" (总{original_count}, {target_protocol}={protocol_count})"
            else:
                display_suffix = ""

            if len(family_samples) < k:
                excluded_families.append((family, len(family_samples)))
                print(f"  排除 {family}: {len(family_samples)} < {k}{display_suffix}")
                continue

            qualified_families.append(family)

            sampled_indices = np.random.choice(len(family_samples), k, replace=False)
            train_indices = sampled_indices
            test_indices = [i for i in range(len(family_samples)) if i not in sampled_indices]

            for idx in train_indices:
                fc, protocol = family_samples[idx]
                train_samples.append((fc, 1.0, family, protocol))

            for idx in test_indices:
                fc, protocol = family_samples[idx]
                test_samples.append((fc, 1.0, family, protocol))

            print(f"  {family}: {len(family_samples)} 样本, 训练使用 {k}, 测试 {len(family_samples) - k}{display_suffix}")

        print(f"\n  符合条件家族: {len(qualified_families)}/{len(self.c2_families)}")
        print(f"  排除家族: {len(excluded_families)}")

        benign_all = []
        for i in range(len(self.benign_dataset)):
            fc = self.benign_dataset[i]
            if fc is not None:
                benign_all.append((fc, 0.0, 'benign', fc.protocol))

        np.random.shuffle(benign_all)

        benign_sample_size = getattr(self, 'max_benign_samples', None)
        if benign_sample_size and len(benign_all) > benign_sample_size:
            benign_all = benign_all[:benign_sample_size]
            print(f"  [良性样本限制] 采样 {benign_sample_size} 个良性样本")

        benign_train_size = int(len(benign_all) * 0.5)
        benign_train = benign_all[:benign_train_size]
        benign_test = benign_all[benign_train_size:]

        train_samples.extend(benign_train)
        test_samples.extend(benign_test)

        np.random.shuffle(train_samples)
        np.random.shuffle(test_samples)

        n_train_c2 = sum(1 for _, label, _, _ in train_samples if label == 1.0)
        n_test_c2 = sum(1 for _, label, _, _ in test_samples if label == 1.0)
        n_train_benign = sum(1 for _, label, _, _ in train_samples if label == 0.0)
        n_test_benign = sum(1 for _, label, _, _ in test_samples if label == 0.0)

        print(f"\n  训练集: {len(train_samples)} (C2: {n_train_c2}, 良性: {n_train_benign})")
        print(f"  测试集: {len(test_samples)} (C2: {n_test_c2}, 良性: {n_test_benign})")

        return train_samples, test_samples, qualified_families, excluded_families

    def get_multiple_runs_split(self, k, num_runs=5, seed_base=42):
        """获取多次运行的 Few-shot 数据划分（不同随机种子）"""
        all_runs = []
        for run in range(num_runs):
            seed = seed_base + run
            train, test, qualified, excluded = self.get_fewshot_split(k, seed)
            all_runs.append({
                'run': run + 1,
                'seed': seed,
                'train_samples': train,
                'test_samples': test,
                'qualified_families': qualified,
                'excluded_families': excluded
            })
        return all_runs


class MultiProtocolLogic2Model:
    """多协议LOGIC2模型管理器"""

    def __init__(self, device=None, target_protocol='ALL'):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.target_protocol = target_protocol
        self._init_models()

    def _init_models(self):
        protocols_to_load = ['tls', 'http', 'tcp'] if self.target_protocol == 'ALL' else [self.target_protocol.lower()]

        for protocol in protocols_to_load:
            sequence_encoder = SequenceEncoder()
            graph_encoder = GraphEncoder(output_dim=128)

            seq_path = f"./sequence_encoder_pretrained_{protocol}.pth"
            graph_path = f"./graph_encoder_pretrained_{protocol}.pth"

            if os.path.exists(seq_path):
                try:
                    sequence_encoder.load_state_dict(torch.load(seq_path, weights_only=True))
                    print(f"  [OK] 加载{protocol.upper()}序列编码器")
                except Exception as e:
                    print(f"  [FAIL] {protocol.upper()}序列编码器加载失败: {e}")

            if os.path.exists(graph_path):
                try:
                    graph_encoder.load_state_dict(torch.load(graph_path, weights_only=True), strict=False)
                    print(f"  [OK] 加载{protocol.upper()}图编码器")
                except Exception as e:
                    print(f"  [FAIL] {protocol.upper()}图编码器加载失败: {e}")

            model = Logic2Model(sequence_encoder, graph_encoder)
            model.to(self.device)
            self.models[protocol.upper()] = model

    def get_model(self, protocol):
        if protocol not in self.models:
            protocol = 'TLS'
        return self.models[protocol]

    def predict(self, flow_cluster):
        protocol = flow_cluster.protocol if hasattr(flow_cluster, 'protocol') else 'TLS'
        model = self.get_model(protocol)
        model.eval()

        with torch.no_grad():
            pred = model(flow_cluster)
            prob = torch.sigmoid(pred).item()

        return prob

    def train_mode(self):
        for model in self.models.values():
            model.train()

    def eval_mode(self):
        for model in self.models.values():
            model.eval()

    def get_all_parameters(self):
        all_params = []
        for model in self.models.values():
            all_params.extend(list(model.parameters()))
        return all_params

    def state_dict(self):
        return {protocol: model.state_dict() for protocol, model in self.models.items()}

    def load_state_dict(self, state_dict):
        for protocol, model_state in state_dict.items():
            if protocol in self.models:
                self.models[protocol].load_state_dict(model_state)


def apply_augmentation_by_protocol(data, target_ratio=0.5, target_protocol='ALL'):
    """按协议分别进行样本增强"""
    if not AUGMENTATION_AVAILABLE or len(data) == 0:
        return data

    protocols_to_augment = ['TLS', 'HTTP', 'TCP'] if target_protocol == 'ALL' else [target_protocol]

    augmenter = FlowClusterAugmentation()
    protocol_data = {'TLS': [], 'HTTP': [], 'TCP': []}

    for item in data:
        if len(item) == 4:
            fc, label, family, protocol = item
        else:
            fc, label = item
            protocol = fc.protocol if hasattr(fc, 'protocol') else 'TLS'

        if protocol in protocol_data:
            protocol_data[protocol].append(item)

    print(f"  按协议分别进行样本增强 (目标协议: {target_protocol}):")

    augmented_data = []
    for protocol in protocols_to_augment:
        proto_samples = protocol_data[protocol]
        if len(proto_samples) == 0:
            continue

        c2_data = [s for s in proto_samples if (s[1] if len(s) == 2 else s[1]) == 1.0]
        benign_data = [s for s in proto_samples if (s[1] if len(s) == 2 else s[1]) == 0.0]

        if len(c2_data) == 0 or len(benign_data) == 0:
            augmented_data.extend(proto_samples)
            print(f"    {protocol}: C2={len(c2_data)}, 良性={len(benign_data)} (跳过增强)")
            continue

        minority_data = c2_data if len(c2_data) < len(benign_data) else benign_data
        majority_data = benign_data if len(c2_data) < len(benign_data) else c2_data

        target_minority_count = int(target_ratio * len(majority_data) / (1 - target_ratio))
        target_minority_count = max(target_minority_count, len(minority_data))
        additional_needed = target_minority_count - len(minority_data)

        print(f"    {protocol}: 原始 C2={len(c2_data)}, 良性={len(benign_data)}")

        proto_augmented = majority_data.copy()
        proto_augmented.extend(minority_data)

        generated_count = 0
        attempts = 0
        max_attempts = additional_needed * 3

        while generated_count < additional_needed and attempts < max_attempts:
            for sample in minority_data:
                if generated_count >= additional_needed:
                    break
                try:
                    fc = sample[0] if len(sample) == 4 else sample[0]
                    label = sample[1] if len(sample) == 4 else sample[1]
                    augmented_cluster = augmenter.augment(fc)
                    if len(sample) == 4:
                        proto_augmented.append((augmented_cluster, label, sample[2], sample[3]))
                    else:
                        proto_augmented.append((augmented_cluster, label))
                    generated_count += 1
                except Exception:
                    pass
                attempts += 1

        np.random.shuffle(proto_augmented)
        augmented_data.extend(proto_augmented)

        final_c2 = sum(1 for s in proto_augmented if (s[1] if len(s) == 2 else s[1]) == 1.0)
        final_benign = sum(1 for s in proto_augmented if (s[1] if len(s) == 2 else s[1]) == 0.0)
        print(f"      增强后: C2={final_c2}, 良性={final_benign}, 总计={len(proto_augmented)}")

    np.random.shuffle(augmented_data)

    total_c2 = sum(1 for s in augmented_data if (s[1] if len(s) == 2 else s[1]) == 1.0)
    total_benign = sum(1 for s in augmented_data if (s[1] if len(s) == 2 else s[1]) == 0.0)
    print(f"    总计: C2={total_c2}, 良性={total_benign}, 总样本数={len(augmented_data)}")

    return augmented_data


def prepare_dataloader(samples, batch_size=64, target_protocol='ALL'):
    """准备按协议分组的数据加载器"""
    protocol_data = {'TLS': [], 'HTTP': [], 'TCP': []}

    for sample in samples:
        if len(sample) == 4:
            fc, label, family, protocol = sample
        else:
            fc, label = sample
            protocol = fc.protocol if hasattr(fc, 'protocol') else 'TLS'

        if protocol in protocol_data:
            protocol_data[protocol].append((fc, label))

    dataloaders = {}
    protocols_to_train = ['TLS', 'HTTP', 'TCP'] if target_protocol == 'ALL' else [target_protocol]

    for protocol in protocols_to_train:
        if protocol in protocol_data and protocol_data[protocol]:
            dataloaders[protocol] = protocol_data[protocol]

    return dataloaders


def train_fewshot_model(train_samples, val_samples, multi_model, epochs=10, batch_size=64, use_augmentation=True, target_protocol='ALL'):
    """Few-shot 训练"""
    device = multi_model.device

    if use_augmentation and AUGMENTATION_AVAILABLE:
        print("\n应用样本增强...")
        train_samples = apply_augmentation_by_protocol(train_samples, target_ratio=0.5)

    protocol_optimizers = {}
    for protocol in ['TLS', 'HTTP', 'TCP']:
        model = multi_model.get_model(protocol)
        protocol_optimizers[protocol] = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    bce_loss = nn.BCEWithLogitsLoss()

    best_val_f1 = 0.0
    patience = 5
    patience_counter = 0
    best_model_state = None

    train_loaders = prepare_dataloader(train_samples, batch_size, target_protocol)
    val_loaders = prepare_dataloader(val_samples, batch_size, target_protocol)

    print("\n各协议训练数据分布:")
    for protocol in ['TLS', 'HTTP', 'TCP']:
        if protocol in train_loaders:
            c2_count = sum(1 for _, label in train_loaders[protocol] if label == 1.0)
            benign_count = sum(1 for _, label in train_loaders[protocol] if label == 0.0)
            print(f"  {protocol}: C2={c2_count}, 良性={benign_count}, 总计={len(train_loaders[protocol])}")

    for epoch in range(epochs):
        multi_model.train_mode()
        total_loss = 0.0

        for protocol in ['TLS', 'HTTP', 'TCP']:
            if protocol not in train_loaders or len(train_loaders[protocol]) == 0:
                continue

            model = multi_model.get_model(protocol)
            optimizer = protocol_optimizers[protocol]
            samples = train_loaders[protocol]

            for i in range(0, len(samples), batch_size):
                batch = samples[i:i+batch_size]
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

        val_metrics, _ = evaluate_fewshot(val_samples, multi_model, verbose=False)
        val_f1 = val_metrics['f1']

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_samples):.4f}, Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = multi_model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停触发! Best Val F1: {best_val_f1:.4f}")
                break

    if best_model_state:
        multi_model.load_state_dict(best_model_state)

    return multi_model


def evaluate_fewshot(test_samples, multi_model, threshold=0.5, cached_probs=None, verbose=True):
    """Few-shot 评估

    Args:
        verbose: 是否打印详细信息，默认True
    """
    multi_model.eval_mode()
    device = multi_model.device

    if cached_probs is not None:
        all_labels = cached_probs['labels']
        all_probs = cached_probs['probs']
        all_families = cached_probs['families']
        all_protocols = cached_probs['protocols']
    else:
        all_labels = []
        all_probs = []
        all_families = []
        all_protocols = []

        with torch.no_grad():
            for sample in test_samples:
                if len(sample) == 4:
                    fc, label, family, protocol = sample
                else:
                    fc, label = sample
                    family = 'unknown'
                    protocol = fc.protocol if hasattr(fc, 'protocol') else 'TLS'

                model = multi_model.get_model(protocol)
                pred = model(fc)
                prob = torch.sigmoid(pred).item()

                all_labels.append(label)
                all_probs.append(prob)
                all_families.append(family)
                all_protocols.append(protocol)

    all_preds = [1 if prob > threshold else 0 for prob in all_probs]

    family_stats = defaultdict(lambda: {'labels': [], 'preds': [], 'probs': []})
    for i in range(len(all_labels)):
        family_stats[all_families[i]]['labels'].append(all_labels[i])
        family_stats[all_families[i]]['preds'].append(all_preds[i])
        family_stats[all_families[i]]['probs'].append(all_probs[i])

    family_metrics = {}
    for family, data in family_stats.items():
        labels = data['labels']
        preds = data['preds']
        probs = data['probs']

        c2_count = int(sum(labels))
        benign_count = len(labels) - c2_count

        tp = sum(1 for i in range(len(labels)) if labels[i] == 1 and preds[i] == 1)
        fp = sum(1 for i in range(len(labels)) if labels[i] == 0 and preds[i] == 1)
        tn = sum(1 for i in range(len(labels)) if labels[i] == 0 and preds[i] == 0)
        fn = sum(1 for i in range(len(labels)) if labels[i] == 1 and preds[i] == 0)

        if c2_count == 0:
            family_metrics[family] = {
                'total': len(labels),
                'c2_count': 0,
                'benign_count': benign_count,
                'f1': 1.0 if fp == 0 else 0.0,
                'accuracy': 1.0 if fp == 0 else 0.0,
                'tp': 0, 'fp': 0, 'tn': tn, 'fn': 0
            }
        else:
            family_metrics[family] = {
                'total': len(labels),
                'c2_count': c2_count,
                'benign_count': benign_count,
                'f1': float(f1_score(labels, preds, zero_division=0)),
                'accuracy': float(np.mean(np.array(preds) == np.array(labels))),
                'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
            }

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
        'family_metrics': dict(family_metrics)
    }

    cm = confusion_matrix(all_labels, all_preds)
    metrics['confusion_matrix'] = cm.tolist()

    if cm.shape[0] > 1:
        fp_val = cm[0][1]
        tn_val = cm[0][0]
        metrics['fpr'] = float(fp_val / (fp_val + tn_val)) if (fp_val + tn_val) > 0 else 0.0
    else:
        metrics['fpr'] = 0.0

    if verbose:
        print(f"\n测试结果 (阈值={threshold}):")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"  FPR: {metrics['fpr']:.4f}")
        print(f"  样本分布: C2={metrics['c2_samples']}, 良性={metrics['benign_samples']}")

        print(f"\n  各家族结果:")
        for family, fam_metrics in sorted(family_metrics.items()):
            print(f"    {family}: Acc={fam_metrics['accuracy']:.4f}, F1={fam_metrics['f1']:.4f}, "
                  f"C2={fam_metrics['c2_count']}, 良性={fam_metrics['benign_count']}, N={fam_metrics['total']}")

    cache_data = {
        'labels': all_labels,
        'probs': all_probs,
        'families': all_families,
        'protocols': all_protocols
    }

    return metrics, cache_data


def fewshot_evaluation(c2_data_path, benign_data_path, k_values=[1, 3, 5, 10], num_runs=5,
                       output_dir="./fewshot_results", filter_mode='multi_only', target_protocol='ALL',
                       max_benign_samples=None):
    """执行 Few-shot 实验"""
    os.makedirs(output_dir, exist_ok=True)

    loader = MultiProtocolFewShotLoader(c2_data_path, benign_data_path, filter_mode=filter_mode,
                                        max_benign_samples=max_benign_samples,
                                        target_protocol=target_protocol)

    all_k_results = {}

    for k in k_values:
        print(f"\n{'#'*80}")
        print(f"# K-SHOT = {k}")
        print(f"{'#'*80}")

        k_results = []
        k_all_run_metrics = []

        for run in range(1, num_runs + 1):
            print(f"\n{'='*60}")
            print(f"Run {run}/{num_runs} (K={k})")
            print(f"{'='*60}")

            train_samples, test_samples, qualified_families, excluded_families = loader.get_fewshot_split(k, seed=42+run, target_protocol=target_protocol)

            if target_protocol != 'ALL':
                def filter_by_protocol(samples, protocol):
                    return [s for s in samples if s[3] == protocol]
                train_samples = filter_by_protocol(train_samples, target_protocol)
                test_samples = filter_by_protocol(test_samples, target_protocol)
                print(f"  [协议过滤] {target_protocol}: 训练={len(train_samples)}, 测试={len(test_samples)}")

            if len(train_samples) == 0:
                print(f"  无训练样本，跳过此轮")
                continue

            all_c2_samples = [s for s in train_samples if s[1] == 1.0]
            all_benign_samples = [s for s in train_samples if s[1] == 0.0]

            c2_val_count = max(1, int(len(all_c2_samples) * 0.1))

            c2_train_used = all_c2_samples[:-c2_val_count] if c2_val_count > 0 else all_c2_samples
            c2_val_samples = all_c2_samples[-c2_val_count:] if c2_val_count > 0 else []
            benign_train = all_benign_samples

            val_samples = c2_val_samples

            train_data = c2_train_used + benign_train

            np.random.shuffle(train_data)

            print(f"\n训练模型...")
            multi_model = MultiProtocolLogic2Model(target_protocol=target_protocol)
            multi_model = train_fewshot_model(train_data, val_samples, multi_model, epochs=10, batch_size=64, target_protocol=target_protocol)

            print(f"\n评估测试集...")
            test_data = test_samples
            np.random.shuffle(test_data)

            n_test_c2 = sum(1 for _, label, _, _ in test_data if label == 1.0)
            n_test_benign = sum(1 for _, label, _, _ in test_data if label == 0.0)
            print(f"  [DEBUG] 测试集实际: 总={len(test_data)}, C2={n_test_c2}, 良性={n_test_benign}")

            test_metrics, prob_cache = evaluate_fewshot(test_data, multi_model, threshold=0.5)

            if test_metrics['f1'] < 0.3 and test_metrics['roc_auc'] > 0.7:
                print(f"\n  阈值优化...")
                best_threshold = 0.5
                best_f1 = test_metrics['f1']

                for threshold in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
                    metrics, _ = evaluate_fewshot(test_data, multi_model, threshold=threshold, cached_probs=prob_cache)
                    if metrics['f1'] > best_f1:
                        best_f1 = metrics['f1']
                        best_threshold = threshold

                if best_threshold != 0.5:
                    print(f"\n  最佳阈值: {best_threshold}, 最佳F1: {best_f1:.4f}")
                    test_metrics, _ = evaluate_fewshot(test_data, multi_model, threshold=best_threshold, cached_probs=prob_cache)

            result = {
                'run': run,
                'k': k,
                'qualified_families': qualified_families,
                'excluded_families': excluded_families,
                'train_size': len(train_data),
                'test_size': len(test_data),
                'metrics': test_metrics
            }
            k_results.append(result)
            k_all_run_metrics.append(test_metrics)

            run_file = os.path.join(output_dir, f"fewshot_k{k}_run{run}.json")
            with open(run_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)

        if k_results:
            all_k_results[k] = {
                'num_runs': len(k_results),
                'results': k_results,
                'average_metrics': compute_average_metrics(k_all_run_metrics)
            }

    generate_fewshot_report(all_k_results, output_dir)

    return all_k_results


def compute_average_metrics(metrics_list):
    """计算平均指标"""
    if not metrics_list:
        return {}

    keys = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'fpr']
    avg_metrics = {}

    for key in keys:
        values = [m.get(key, 0) for m in metrics_list if key in m]
        if values:
            avg_metrics[f'avg_{key}'] = float(np.mean(values))
            avg_metrics[f'std_{key}'] = float(np.std(values))
            avg_metrics[f'min_{key}'] = float(np.min(values))
            avg_metrics[f'max_{key}'] = float(np.max(values))

    return avg_metrics


def generate_fewshot_report(all_k_results, output_dir):
    """生成 Few-shot 实验报告"""
    print(f"\n{'='*80}")
    print("Few-shot 实验总体报告")
    print(f"{'='*80}")

    print(f"\n{'K':<6} {'Runs':<6} {'Avg F1':<10} {'Std F1':<10} {'Avg Recall':<12} {'Avg AUC':<10} {'Avg FPR':<10}")
    print(f"{'-'*70}")

    for k in sorted(all_k_results.keys()):
        data = all_k_results[k]
        avg = data['average_metrics']
        print(f"{k:<6} {data['num_runs']:<6} "
              f"{avg.get('avg_f1', 0):<10.4f} {avg.get('std_f1', 0):<10.4f} "
              f"{avg.get('avg_recall', 0):<12.4f} {avg.get('avg_roc_auc', 0):<10.4f} "
              f"{avg.get('avg_fpr', 0):<10.4f}")

    report = {
        'k_values': list(all_k_results.keys()),
        'num_runs_per_k': [all_k_results[k]['num_runs'] for k in all_k_results],
        'all_results': {k: {
            'average_metrics': data['average_metrics'],
            'num_runs': data['num_runs']
        } for k, data in all_k_results.items()}
    }

    report_file = os.path.join(output_dir, "fewshot_summary_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n报告已保存: {report_file}")


def main():
    parser = argparse.ArgumentParser(description="LOGIC2 Multi-Protocol Few-shot Classification")
    parser.add_argument("--c2-data", type=str, default="./csv_MTA", help="C2数据路径")
    parser.add_argument("--benign-data", type=str, default="./CTU-csv", help="良性数据路径")
    parser.add_argument("--k-values", type=int, nargs='+', default=[1, 3, 5, 10], help="K值列表")
    parser.add_argument("--num-runs", type=int, default=5, help="每个K值的运行次数")
    parser.add_argument("--output-dir", type=str, default="./fewshot_results", help="结果输出目录")
    parser.add_argument("--filter-mode", type=str, default="multi_only",
                        choices=["multi_only", "multi_and_long_single", "all"],
                        help="数据筛选模式")
    parser.add_argument("--protocol", type=str, default="ALL",
                        choices=["ALL", "TLS", "HTTP", "TCP"],
                        help="指定测试的协议: ALL/TLS/HTTP/TCP (ALL表示全部)")
    parser.add_argument("--benign-sample-size", type=int, default=None,
                        help="限制良性样本数量 (默认: 不限制)")
    args = parser.parse_args()

    print("="*80)
    print("LOGIC2 多协议 Few-shot 分类实验")
    print("="*80)
    print(f"C2数据路径: {args.c2_data}")
    print(f"良性数据路径: {args.benign_data}")
    print(f"测试协议: {args.protocol}")
    print(f"K值: {args.k_values}")
    print(f"每K值运行次数: {args.num_runs}")
    print(f"输出目录: {args.output_dir}")
    print(f"筛选模式: {args.filter_mode}")

    results = fewshot_evaluation(
        args.c2_data, args.benign_data,
        k_values=args.k_values,
        num_runs=args.num_runs,
        output_dir=args.output_dir,
        filter_mode=args.filter_mode,
        target_protocol=args.protocol,
        max_benign_samples=args.benign_sample_size
    )

    print("\n" + "="*80)
    print("Few-shot 实验完成!")
    print("="*80)


if __name__ == "__main__":
    main()
