"""
数据增强脚本
用于平衡两个数据集（ C2 和 良性）中的 train 数据

使用方法:
    python data_augmentation.py --c2-data <C2数据集路径> --benign-data <良性数据集路径>

数据集目录结构:
    <根目录>/
        http/
            train/  (csv文件)
            test/   (csv文件，不参与平衡)
        tls/
            train/  (csv文件)
            test/   (csv文件，不参与平衡)
        tcp/
            train/  (csv文件)
            test/   (csv文件，不参与平衡)
"""

import os
import sys
import csv
import copy
import argparse
import numpy as np
from collections import defaultdict
from datetime import datetime


class FlowBlock:
    def __init__(self, direction, payload_len, time_interval, timestamp):
        self.direction = direction
        self.payload_len = payload_len
        self.time_interval = time_interval
        self.timestamp = timestamp


class Flow:
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
    def __init__(self, src_ip, dst_ip, dst_port, protocol, filename=None):
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.dst_port = dst_port
        self.protocol = protocol
        self.filename = filename
        self.flows = []
        self.logical_sequence = []
        self.graph = None


class FlowClusterAugmentation:
    def __init__(self, drop_short_flow_prob=0.2, min_cluster_size_ratio=0.8,
                 time_jitter_range=0.5, mask_block_prob=0.05):
        self.drop_short_flow_prob = drop_short_flow_prob
        self.min_cluster_size_ratio = min_cluster_size_ratio
        self.time_jitter_range = time_jitter_range
        self.mask_block_prob = mask_block_prob

    def augment(self, flow_cluster):
        augmented_cluster = copy.deepcopy(flow_cluster)
        augmented_cluster = self._random_flow_drop(augmented_cluster)
        augmented_cluster = self._flow_temporal_jitter(augmented_cluster)
        return augmented_cluster

    def _random_flow_drop(self, flow_cluster):
        if np.random.rand() > self.drop_short_flow_prob:
            return flow_cluster

        long_flows = [flow for flow in flow_cluster.flows if flow.is_long_flow]
        short_flows = [flow for flow in flow_cluster.flows if not flow.is_long_flow]

        min_size = int(len(flow_cluster.flows) * self.min_cluster_size_ratio)
        min_size = max(min_size, len(long_flows))

        max_drop = len(short_flows) - (min_size - len(long_flows))
        max_drop = max(max_drop, 0)

        if max_drop > 0:
            np.random.shuffle(short_flows)
            short_flows = short_flows[:-max_drop] if max_drop < len(short_flows) else []

        flow_cluster.flows = long_flows + short_flows

        all_blocks = []
        for flow in flow_cluster.flows:
            for block in flow.blocks:
                all_blocks.append((block.timestamp, block))
        all_blocks.sort(key=lambda x: x[0])
        flow_cluster.logical_sequence = [block for _, block in all_blocks]

        return flow_cluster

    def _flow_temporal_jitter(self, flow_cluster):
        for flow in flow_cluster.flows:
            original_blocks = copy.deepcopy(flow.blocks)
            new_blocks = []

            for i, block in enumerate(original_blocks):
                if np.random.rand() < self.mask_block_prob:
                    continue

                jittered_block = copy.deepcopy(block)

                if i > 0:
                    jitter = 1.0 + np.random.uniform(-self.time_jitter_range, self.time_jitter_range)
                    jittered_block.time_interval = max(0, int(jittered_block.time_interval * jitter))

                new_blocks.append(jittered_block)

            flow.blocks = new_blocks

            if new_blocks:
                current_time = new_blocks[0].timestamp
                for i, block in enumerate(new_blocks):
                    if i == 0:
                        block.timestamp = current_time
                    else:
                        current_time += block.time_interval
                        block.timestamp = current_time

                flow.duration = new_blocks[-1].timestamp - new_blocks[0].timestamp

        all_blocks = []
        for flow in flow_cluster.flows:
            for block in flow.blocks:
                all_blocks.append((block.timestamp, block))
        all_blocks.sort(key=lambda x: x[0])
        flow_cluster.logical_sequence = [block for _, block in all_blocks]

        return flow_cluster


def load_csv_to_flow_cluster(csv_file, protocol):
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return None

    first_row = rows[0]
    src_ip = first_row['Source IP']
    dst_ip = first_row['Destination IP']
    dst_port = int(first_row['Destination Port'])

    filename = os.path.basename(csv_file)
    flow_cluster = FlowCluster(src_ip, dst_ip, dst_port, protocol, filename)

    session_dict = defaultdict(list)
    for row in rows:
        session_id = int(row['Session ID'])
        session_dict[session_id].append(row)

    for session_id, session_rows in session_dict.items():
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

    flow_cluster.logical_sequence = []
    all_blocks = []
    for flow in flow_cluster.flows:
        for block in flow.blocks:
            all_blocks.append((block.timestamp, block))
    all_blocks.sort(key=lambda x: x[0])
    flow_cluster.logical_sequence = [block for _, block in all_blocks]

    return flow_cluster


def save_flow_cluster_to_csv(flow_cluster, output_file):
    rows = []

    for flow in flow_cluster.flows:
        session_id = flow.session_id
        session_duration = flow.duration

        for block in flow.blocks:
            row = {
                'Session ID': session_id,
                'Length': block.payload_len if block.direction == 0 else -block.payload_len,
                'Timestamp': block.timestamp,
                'Session Duration': session_duration,
                'Source IP': flow.src_ip,
                'Destination IP': flow.dst_ip,
                'Source Port': flow.src_port,
                'Destination Port': flow.dst_port,
                'Protocols': flow.protocol.lower(),
                'Type': 'block',
                'Packet Count': len(flow.blocks),
                'Block Count': len(flow_cluster.flows),
                'Protocol': flow.protocol.lower(),
                'TLS Version': ''
            }
            rows.append(row)

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


def collect_samples(data_dir, protocol):
    train_dir = os.path.join(data_dir, protocol, 'train')
    if not os.path.exists(train_dir):
        return []

    samples = []
    for filename in os.listdir(train_dir):
        if filename.endswith('.csv'):
            csv_path = os.path.join(train_dir, filename)
            flow_cluster = load_csv_to_flow_cluster(csv_path, protocol.upper())
            if flow_cluster is not None:
                samples.append((flow_cluster, filename, csv_path))

    return samples


def balance_train_data(c2_dir, benign_dir, protocols=['http', 'tls', 'tcp']):
    augmenter = FlowClusterAugmentation()

    for protocol in protocols:
        print(f"\n{'='*60}")
        print(f"处理协议: {protocol.upper()}")
        print('='*60)

        c2_samples = collect_samples(c2_dir, protocol)
        benign_samples = collect_samples(benign_dir, protocol)

        c2_count = len(c2_samples)
        benign_count = len(benign_samples)

        print(f"  C2 样本数: {c2_count}")
        print(f"  良性样本数: {benign_count}")

        if c2_count == 0 or benign_count == 0:
            print(f"  跳过平衡（某类样本为空）")
            continue

        minority_samples = c2_samples if c2_count < benign_count else benign_samples
        majority_samples = benign_samples if c2_count < benign_count else c2_samples
        minority_label = "C2" if c2_count < benign_count else "良性"

        augmentation_times = len(majority_samples) // len(minority_samples) if len(minority_samples) > 0 else 0
        augmentation_times = max(augmentation_times, 1)

        print(f"  少数类: {minority_label}")
        print(f"  多数类样本数: {len(majority_samples)}")
        print(f"  少数类样本数: {len(minority_samples)}")
        print(f"  增强倍数: {augmentation_times}x")

        augmented_dir = os.path.join(c2_dir if minority_label == "C2" else benign_dir, protocol, 'train_augmented')
        os.makedirs(augmented_dir, exist_ok=True)

        for flow_cluster, filename, original_path in minority_samples:
            save_flow_cluster_to_csv(flow_cluster, os.path.join(augmented_dir, filename))

        generated = 0
        for flow_cluster, _, _ in minority_samples:
            for i in range(augmentation_times):
                np.random.seed(None)
                augmented = augmenter.augment(flow_cluster)

                base_name = flow_cluster.filename or f"augmented_{generated}"
                name_parts = base_name.split('.')
                if len(name_parts) > 1:
                    aug_filename = f"aug_{i}_{generated}." + ".".join(name_parts[1:])
                else:
                    aug_filename = f"aug_{i}_{generated}.csv"

                save_flow_cluster_to_csv(augmented, os.path.join(augmented_dir, aug_filename))
                generated += 1

        final_minority_count = len([f for f in os.listdir(augmented_dir) if f.endswith('.csv')])

        print(f"  增强完成!")
        print(f"  原始少数类: {len(minority_samples)}")
        print(f"  增强后少数类 (train_augmented): {final_minority_count}")


def main():
    parser = argparse.ArgumentParser(
        description="数据增强脚本 - 平衡 C2 和 良性 数据集的 train 数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python data_augmentation.py --c2-data ./C2-data --benign-data ./Benign-data

数据集目录结构:
    <根目录>/
        http/
            train/  (csv文件)
            test/   (csv文件，不参与平衡)
        tls/
            train/  (csv文件)
            test/   (csv文件，不参与平衡)
        tcp/
            train/  (csv文件)
            test/   (csv文件，不参与平衡)
        """
    )

    parser.add_argument('--c2-data', type=str, required=True,
                        help='C2数据集根目录路径')
    parser.add_argument('--benign-data', type=str, required=True,
                        help='良性数据集根目录路径')
    parser.add_argument('--target-ratio', type=float, default=0.5,
                        help='目标少数类比例 (默认: 0.5，即平衡)')
    parser.add_argument('--protocols', type=str, nargs='+',
                        default=['http', 'tls', 'tcp'],
                        help='要处理的协议列表 (默认: http tls tcp)')

    args = parser.parse_args()

    if not os.path.exists(args.c2_data):
        print(f"错误: C2数据目录不存在: {args.c2_data}")
        sys.exit(1)

    if not os.path.exists(args.benign_data):
        print(f"错误: 良性数据目录不存在: {args.benign_data}")
        sys.exit(1)

    print("="*60)
    print("数据增强脚本")
    print("="*60)
    print(f"C2 数据目录: {args.c2_data}")
    print(f"良性数据目录: {args.benign_data}")
    print(f"目标少数类比例: {args.target_ratio}")
    print(f"处理的协议: {', '.join(args.protocols)}")
    print("="*60)

    balance_train_data(args.c2_data, args.benign_data, args.protocols)

    print("\n" + "="*60)
    print("所有协议处理完成!")
    print("="*60)


if __name__ == "__main__":
    main()
