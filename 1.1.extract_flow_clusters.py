"""
PCAP流簇提取工具（优化版）
将PCAP文件按流簇（四元组）拆分成多个子PCAP文件
每个子PCAP文件包含一个流簇的所有数据包
优化点：提取全局服务器特征，优先匹配减少重复判断，提升准确性和效率
"""

import os
import subprocess
import argparse
import logging
from collections import defaultdict
from scapy.all import rdpcap, wrpcap, IP, TCP, Raw
import tempfile
import shutil

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def extract_servers_with_tshark(pcap_file):
    """
    使用 tshark 提取 PCAP 文件中的 HTTP 和 TLS 服务器信息
    """
    server_set = set()
    
    # 提取 HTTP 服务器信息
    http_command = [
        'tshark',
        '-r', pcap_file,
        '-Y', 'http',
        '-T', 'fields',
        '-e', 'ip.src',
        '-e', 'tcp.srcport',
        '-e', 'ip.dst',
        '-e', 'tcp.dstport',
        '-e', 'http.request.method',
        '-e', 'http.response.code'
    ]
    
    # 提取 TLS 服务器信息
    tls_command = [
        'tshark',
        '-r', pcap_file,
        '-Y', 'tls.handshake.type == 1 or tls.handshake.type == 2',
        '-T', 'fields',
        '-e', 'ip.src',
        '-e', 'tcp.srcport',
        '-e', 'ip.dst',
        '-e', 'tcp.dstport',
        '-e', 'tls.handshake.type'
    ]
    
    # 执行 HTTP 分析
    try:
        http_output = subprocess.check_output(http_command, text=True)
        lines = http_output.strip().split('\n')
        for line in lines:
            if not line:
                continue
            parts = line.split('\t')
            while len(parts) < 6:
                parts.append('')
            src_ip, src_port, dst_ip, dst_port, method, response = parts[:6]
            
            if method:
                # HTTP 请求：目标是服务器
                server_ip = dst_ip
                server_port = dst_port
                server_set.add((server_ip, int(server_port)))
            elif response:
                # HTTP 响应：源是服务器
                server_ip = src_ip
                server_port = src_port
                server_set.add((server_ip, int(server_port)))
    except subprocess.CalledProcessError:
        pass
    
    # 执行 TLS 分析
    try:
        tls_output = subprocess.check_output(tls_command, text=True)
        lines = tls_output.strip().split('\n')
        for line in lines:
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 5:
                continue
            src_ip, src_port, dst_ip, dst_port, handshake_type = parts[:5]
            
            if handshake_type == '1':
                # ClientHello：目标是服务器
                server_ip = dst_ip
                server_port = dst_port
                server_set.add((server_ip, int(server_port)))
            elif handshake_type == '2':
                # ServerHello：源是服务器
                server_ip = src_ip
                server_port = src_port
                server_set.add((server_ip, int(server_port)))
    except subprocess.CalledProcessError:
        pass
    
    return server_set


def is_private_ip(ip):
    """
    判断是否为内网IP地址
    内网IP范围：
    - 10.0.0.0/8 (10.0.0.0 - 10.255.255.255)
    - 172.16.0.0/12 (172.16.0.0 - 172.31.255.255)
    - 192.168.0.0/16 (192.168.0.0 - 192.168.255.255)
    """
    parts = list(map(int, ip.split('.')))
    if len(parts) != 4:
        return False
    
    # 10.0.0.0/8
    if parts[0] == 10:
        return True
    # 172.16.0.0/12
    if parts[0] == 172 and 16 <= parts[1] <= 31:
        return True
    # 192.168.0.0/16
    if parts[0] == 192 and parts[1] == 168:
        return True
    
    return False


def is_http_request(packet):
    """
    检测是否是HTTP请求报文
    返回: True/False
    """
    if not packet.haslayer(Raw):
        return False
    
    payload = bytes(packet[Raw].load)
    
    try:
        payload_str = payload[:20].decode('utf-8', errors='ignore').upper()
        # HTTP请求方法
        if any(payload_str.startswith(method) for method in ['GET ', 'POST ', 'PUT ', 'DELETE ', 'HEAD ', 'OPTIONS ', 'PATCH ', 'CONNECT ']):
            return True
    except:
        pass
    
    return False


def is_http_response(packet):
    """
    检测是否是HTTP响应报文
    返回: True/False
    """
    if not packet.haslayer(Raw):
        return False
    
    payload = bytes(packet[Raw].load)
    
    try:
        payload_str = payload[:20].decode('utf-8', errors='ignore').upper()
        # HTTP响应以 "HTTP/1." 或 "HTTP/2." 开头
        if payload_str.startswith('HTTP/1.') or payload_str.startswith('HTTP/2.'):
            return True
    except:
        pass
    
    return False


def is_tls_client_hello(packet):
    """
    检测是否是TLS ClientHello报文
    返回: True/False
    """
    if not packet.haslayer(Raw):
        return False
    
    payload = bytes(packet[Raw].load)
    
    # TLS记录格式: 类型(1) + 版本(2) + 长度(2) + ...
    # Handshake类型22，ClientHello类型1
    if len(payload) >= 6:
        if payload[0] == 22:  # Handshake
            if payload[5] == 1:  # ClientHello
                return True
    
    return False


def is_tls_server_hello(packet):
    """
    检测是否是TLS ServerHello报文
    返回: True/False
    """
    if not packet.haslayer(Raw):
        return False
    
    payload = bytes(packet[Raw].load)
    
    # TLS记录格式: 类型(1) + 版本(2) + 长度(2) + ...
    # Handshake类型22，ServerHello类型2
    if len(payload) >= 6:
        if payload[0] == 22:  # Handshake
            if payload[5] == 2:  # ServerHello
                return True
    
    return False


def get_flow_key(packet):
    """
    获取数据包的流簇标识（四元组）
    流簇定义：同一客户端IP访问同一服务器IP的同一端口的所有流量
    
    判断服务器的逻辑：
    1. HTTP请求：源IP是客户端，目标IP是服务器
    2. HTTP响应：源IP是服务器，目标IP是客户端
    3. TLS ClientHello：源IP是客户端，目标IP是服务器
    4. 内网IP通常是客户端，公网IP通常是服务器
    5. 较小的端口通常是服务器端口（知名端口）
    """
    if not packet.haslayer(IP) or not packet.haslayer(TCP):
        return None
    
    ip_layer = packet[IP]
    tcp_layer = packet[TCP]
    
    src_ip = ip_layer.src
    dst_ip = ip_layer.dst
    src_port = tcp_layer.sport
    dst_port = tcp_layer.dport
    
    # 优先通过请求报文判断（HTTP和TLS）
    if is_http_request(packet):
        # HTTP请求：源IP是客户端，目标IP是服务器
        client_ip = src_ip
        server_ip = dst_ip
        server_port = dst_port
        return (client_ip, server_ip, server_port)
    
    # 通过HTTP响应报文判断
    if is_http_response(packet):
        # HTTP响应：源IP是服务器，目标IP是客户端
        client_ip = dst_ip
        server_ip = src_ip
        server_port = src_port
        return (client_ip, server_ip, server_port)
    
    if is_tls_client_hello(packet):
        # TLS ClientHello：源IP是客户端，目标IP是服务器
        client_ip = src_ip
        server_ip = dst_ip
        server_port = dst_port
        return (client_ip, server_ip, server_port)
    
    # 判断内网IP
    src_is_private = is_private_ip(src_ip)
    dst_is_private = is_private_ip(dst_ip)
    
    # 通过内网IP判断客户端
    if src_is_private and not dst_is_private:
        # 源IP是内网，目标IP是公网 -> 源是客户端
        client_ip = src_ip
        server_ip = dst_ip
        server_port = dst_port
    elif dst_is_private and not src_is_private:
        # 目标IP是内网，源IP是公网 -> 目标是客户端
        client_ip = dst_ip
        server_ip = src_ip
        server_port = src_port
    else:
        # 两个都是内网或两个都是公网，通过端口判断
        # 注意：在WebShell场景中，服务器可能运行在高端口
        # 优先检查常见服务器端口，然后再考虑端口大小
        common_server_ports = [80, 443, 8080, 8443, 7001, 9000, 9001, 8000]
        
        # 检查是否是常见服务器端口
        if dst_port in common_server_ports:
            # 目标端口是常见服务器端口，目标是服务器
            server_ip = dst_ip
            server_port = dst_port
            client_ip = src_ip
        elif src_port in common_server_ports:
            # 源端口是常见服务器端口，源是服务器
            server_ip = src_ip
            server_port = src_port
            client_ip = dst_ip
        else:
            # 较小的端口通常是服务器端口
            if src_port < dst_port:
                server_ip = src_ip
                server_port = src_port
                client_ip = dst_ip
            else:
                server_ip = dst_ip
                server_port = dst_port
                client_ip = src_ip
    
    # 四元组：(客户端IP, 服务器IP, 服务器端口)
    return (client_ip, server_ip, server_port)


def detect_protocol(packet):
    """
    检测数据包的协议类型
    返回: 'tls', 'http', 'tcp'
    """
    if not packet.haslayer(Raw):
        return 'tcp'
    
    payload = bytes(packet[Raw].load)
    
    # 检测HTTP（优先检测，因为HTTP更容易识别）
    try:
        payload_str = payload[:100].decode('utf-8', errors='ignore').upper()
        # HTTP请求方法
        if any(payload_str.startswith(method) for method in ['GET ', 'POST ', 'PUT ', 'DELETE ', 'HEAD ', 'OPTIONS ', 'PATCH ', 'CONNECT ']):
            return 'http'
        # HTTP响应
        if payload_str.startswith('HTTP/1.') or payload_str.startswith('HTTP/2.'):
            return 'http'
        # HTTP头部特征
        if any(header in payload_str for header in ['HOST:', 'USER-AGENT:', 'CONTENT-TYPE:', 'CONTENT-LENGTH:', 'CONNECTION:']):
            return 'http'
    except:
        pass
    
    # 检测TLS（更严格的检测）
    if len(payload) >= 6:
        first_byte = payload[0]
        # TLS记录类型: 20(ChangeCipherSpec), 21(Alert), 22(Handshake), 23(Application)
        if first_byte in [20, 21, 22, 23]:
            # 额外检查：TLS记录长度字段应该合理
            tls_record_length = (payload[3] << 8) | payload[4]
            # TLS记录长度应该在合理范围内（0-16384）
            if 0 < tls_record_length <= 16384:
                # 检查版本号
                version = (payload[1], payload[2])
                if version in [(3, 1), (3, 2), (3, 3), (3, 4)]:  # TLS 1.0, 1.1, 1.2, 1.3
                    return 'tls'
    
    return 'tcp'


def determine_flow_protocol(packets):
    """
    确定流簇的主要协议类型
    优先级: TLS > HTTP > TCP
    """
    tls_count = 0
    http_count = 0
    
    for pkt in packets:
        proto = detect_protocol(pkt)
        if proto == 'tls':
            tls_count += 1
        elif proto == 'http':
            http_count += 1
    
    if tls_count > 0:
        return 'tls'
    elif http_count > 0:
        return 'http'
    else:
        return 'tcp'


def is_tls_handshake(packet):
    """
    判断是否为TLS握手阶段的数据包
    TLS握手类型: 1=ClientHello, 2=ServerHello, 11=Certificate, 12=ServerKeyExchange,
                13=CertificateRequest, 14=ServerHelloDone, 15=CertificateVerify,
                16=ClientKeyExchange, 20=Finished
    """
    if not packet.haslayer(Raw):
        return False
    
    try:
        payload = bytes(packet[Raw].load)
        if len(payload) < 6:
            return False
        
        # TLS记录类型
        record_type = payload[0]
        
        # Handshake记录类型为22
        if record_type == 22:
            # 检查是否是握手消息
            if len(payload) >= 6:
                # 跳过TLS记录头（5字节），获取握手类型
                handshake_type = payload[5]
                # 握手类型 1-20 都是握手阶段
                if handshake_type in [1, 2, 3, 4, 11, 12, 13, 14, 15, 16, 20]:
                    return True
        
        # ChangeCipherSpec (20) 也属于握手阶段
        if record_type == 20:
            return True
        
        # Alert (21) 可能是握手失败
        if record_type == 21:
            return True
            
    except:
        pass
    
    return False


def is_tcp_handshake(packet):
    """
    判断是否为TCP握手阶段的数据包
    TCP握手: SYN, SYN-ACK, ACK (三次握手)
    """
    if not packet.haslayer(TCP):
        return False
    
    tcp_layer = packet[TCP]
    flags = tcp_layer.flags
    
    # SYN=0x02, SYN-ACK=0x12, ACK=0x10 (纯ACK)
    # 如果是SYN或SYN-ACK，属于握手阶段
    if flags & 0x02:  # SYN flag
        return True
    
    # 纯ACK且没有数据载荷，可能是握手或keepalive
    if flags == 0x10 and not packet.haslayer(Raw):
        return True
    
    return False


def has_valid_payload(packet):
    """
    判断数据包是否有有效的TCP载荷
    有效载荷：握手阶段之外，且载荷长度>0
    """
    # 检查是否有TCP层
    if not packet.haslayer(TCP):
        return False
    
    # 检查是否是TCP握手阶段
    if is_tcp_handshake(packet):
        return False
    
    # 检查是否有载荷
    if not packet.haslayer(Raw):
        return False
    
    payload = bytes(packet[Raw].load)
    
    # 载荷长度为0
    if len(payload) == 0:
        return False
    
    # 检查是否是TLS握手阶段
    if is_tls_handshake(packet):
        return False
    
    return True


def flow_has_valid_packets(packets):
    """
    判断流簇是否有有效数据包
    有效数据包：握手阶段之外，TCP载荷长度>0
    """
    for pkt in packets:
        if has_valid_payload(pkt):
            return True
    return False


def extract_flows_from_pcap(pcap_file, output_folder, input_folder=None):
    """
    从PCAP文件中提取所有流簇并保存为单独的PCAP文件（优化版）
    新增：提取全局服务器特征，优先匹配减少重复判断
    
    Args:
        pcap_file: 输入的PCAP文件路径
        output_folder: 输出文件夹路径
        input_folder: 输入根文件夹路径，用于保持目录结构
    """
    logging.info(f"处理文件: {pcap_file}")
    
    # 读取PCAP文件
    try:
        packets = rdpcap(pcap_file)
    except Exception as e:
        logging.error(f"无法读取PCAP文件 {pcap_file}: {e}")
        return 0
    
    if len(packets) == 0:
        logging.warning(f"PCAP文件为空: {pcap_file}")
        return 0
    
    # ===================== 优化核心：第一步 - 提取全局服务器特征 =====================
    # 全局服务器集合：{(server_ip, server_port), ...}
    # 优先使用 tshark 提取，提高准确性
    global_server_set = set()
    
    # 尝试使用 tshark 提取服务器信息
    tshark_servers = extract_servers_with_tshark(pcap_file)
    if tshark_servers:
        global_server_set = tshark_servers
        logging.info(f"使用 tshark 提取到全局服务器特征: {global_server_set}")
    else:
        # 如果 tshark 提取失败，使用基于 scapy 的方法
        for pkt in packets:
            if not pkt.haslayer(IP) or not pkt.haslayer(TCP):
                continue
            
            ip_layer = pkt[IP]
            tcp_layer = pkt[TCP]
            
            # 从HTTP请求提取服务器信息
            if is_http_request(pkt):
                server_ip = ip_layer.dst
                server_port = tcp_layer.dport
                # 检查端口是否合理（常见服务器端口或知名端口）
                if server_port in [80, 443, 8080, 8443] or (1 <= server_port <= 1024):
                    global_server_set.add((server_ip, server_port))
            # 从HTTP响应提取服务器信息
            elif is_http_response(pkt):
                server_ip = ip_layer.src
                server_port = tcp_layer.sport
                # 检查端口是否合理（常见服务器端口或知名端口）
                if server_port in [80, 443, 8080, 8443] or (1 <= server_port <= 1024):
                    global_server_set.add((server_ip, server_port))
            # 从TLS ClientHello提取服务器信息
            elif is_tls_client_hello(pkt):
                server_ip = ip_layer.dst
                server_port = tcp_layer.dport
                # 检查端口是否合理（常见服务器端口或知名端口）
                if server_port in [80, 443, 8080, 8443] or (1 <= server_port <= 1024):
                    global_server_set.add((server_ip, server_port))
        
        if global_server_set:
            logging.info(f"使用 scapy 提取到全局服务器特征: {global_server_set}")
    
    if global_server_set:
        logging.info(f"提取到全局服务器特征: {global_server_set}")
    else:
        logging.info("未提取到明确的全局服务器特征，将使用兜底逻辑")
    
    # ===================== 第二步 - 建立流-服务器映射（优化版） =====================
    flow_server_map = {}
    
    for pkt in packets:
        if not pkt.haslayer(IP) or not pkt.haslayer(TCP):
            continue
        
        src_ip = pkt[IP].src
        dst_ip = pkt[IP].dst
        src_port = pkt[TCP].sport
        dst_port = pkt[TCP].dport
        
        # 双向流标识（规范化）
        # 先比较IP地址的数值大小，再比较端口
        def ip_to_tuple(ip):
            return tuple(map(int, ip.split('.')))
        
        src_ip_tuple = ip_to_tuple(src_ip)
        dst_ip_tuple = ip_to_tuple(dst_ip)
        
        if (src_ip_tuple, src_port) < (dst_ip_tuple, dst_port):
            flow_id = (src_ip, src_port, dst_ip, dst_port)
        else:
            flow_id = (dst_ip, dst_port, src_ip, src_port)
        
        # 如果已经确定服务器，跳过
        if flow_id in flow_server_map:
            continue
        
        # 优先匹配全局服务器特征
        client_ip = None
        server_ip = None
        server_port = None
        
        # 检查当前流的两端是否匹配全局服务器
        if (src_ip, src_port) in global_server_set:
            # 源端是服务器
            server_ip = src_ip
            server_port = src_port
            client_ip = dst_ip
        elif (dst_ip, dst_port) in global_server_set:
            # 目的端是服务器
            server_ip = dst_ip
            server_port = dst_port
            client_ip = src_ip
        
        # 如果匹配到全局服务器，直接生成流簇标识
        if client_ip and server_ip and server_port:
            flow_key = (client_ip, server_ip, server_port)
            flow_server_map[flow_id] = flow_key
            continue
        
        # 如果未匹配到，再使用原有的get_flow_key逻辑（兜底）
        flow_key = get_flow_key(pkt)
        if flow_key:
            flow_server_map[flow_id] = flow_key
    
    # ===================== 第三步 - 按流簇分组数据包 =====================
    flows = defaultdict(list)
    
    for pkt in packets:
        if not pkt.haslayer(IP) or not pkt.haslayer(TCP):
            continue
        
        src_ip = pkt[IP].src
        dst_ip = pkt[IP].dst
        src_port = pkt[TCP].sport
        dst_port = pkt[TCP].dport
        
        # 双向流标识
        # 先比较IP地址的数值大小，再比较端口
        def ip_to_tuple(ip):
            return tuple(map(int, ip.split('.')))
        
        src_ip_tuple = ip_to_tuple(src_ip)
        dst_ip_tuple = ip_to_tuple(dst_ip)
        
        if (src_ip_tuple, src_port) < (dst_ip_tuple, dst_port):
            flow_id = (src_ip, src_port, dst_ip, dst_port)
        else:
            flow_id = (dst_ip, dst_port, src_ip, src_port)
        
        # 查找服务器映射
        if flow_id in flow_server_map:
            flow_key = flow_server_map[flow_id]
        else:
            # 最后的备选方案：先查全局服务器，再用端口判断
            if (src_ip, src_port) in global_server_set:
                flow_key = (dst_ip, src_ip, src_port)
            elif (dst_ip, dst_port) in global_server_set:
                flow_key = (src_ip, dst_ip, dst_port)
            elif src_port < dst_port:
                flow_key = (dst_ip, src_ip, src_port)
            else:
                flow_key = (src_ip, dst_ip, dst_port)
        
        flows[flow_key].append(pkt)
    
    # ===================== 第四步 - 保存流簇文件 =====================
    # 获取原始文件名（不含扩展名）
    base_name = os.path.splitext(os.path.basename(pcap_file))[0]
    
    # 计算相对路径，保持目录结构
    if input_folder:
        rel_path = os.path.relpath(os.path.dirname(pcap_file), input_folder)
        if rel_path == '.':
            rel_path = ''
    else:
        rel_path = ''
    
    # 构建输出子文件夹路径
    pcap_output_folder = os.path.join(output_folder, rel_path)
    os.makedirs(pcap_output_folder, exist_ok=True)
    
    # 统计信息
    stats = {'tls': 0, 'http': 0, 'tcp': 0, 'skipped': 0}
    
    # 保存每个流簇
    for idx, (flow_key, flow_packets) in enumerate(flows.items(), 1):
        # 检查是否有有效数据包
        if not flow_has_valid_packets(flow_packets):
            stats['skipped'] += 1
            continue
        
        client_ip, server_ip, server_port = flow_key
        
        # 确定协议类型
        protocol = determine_flow_protocol(flow_packets)
        stats[protocol] += 1
        
        # 生成文件名，保留原始文件名前缀（截断到固定长度）
        original_base = os.path.splitext(os.path.basename(pcap_file))[0]
        # 截断到100个字符，避免路径过长
        max_prefix_len = 100
        if len(original_base) > max_prefix_len:
            prefix = original_base[:max_prefix_len]
        else:
            prefix = original_base
        
        # 生成文件名: 原始文件名前缀_流序号_协议.pcap
        flow_filename = f"{prefix}_{idx:03d}_{protocol}.pcap"
        output_path = os.path.join(pcap_output_folder, flow_filename)
        
        # 检查路径长度，如果超过限制则使用更短的文件名
        if len(output_path) > 250:
            flow_filename = f"{prefix[:15]}_{idx:03d}_{protocol}.pcap"
            output_path = os.path.join(pcap_output_folder, flow_filename)
        
        # 保存PCAP文件
        try:
            wrpcap(output_path, flow_packets)
            logging.info(f"  保存流簇 {idx}: {len(flow_packets)} 个包, 协议={protocol}, 文件={flow_filename}")
        except Exception as e:
            logging.error(f"  保存流簇 {idx} 失败: {e}")
    
    logging.info(f"完成 {pcap_file}: TLS={stats['tls']}, HTTP={stats['http']}, TCP={stats['tcp']}, 跳过(无有效数据)={stats['skipped']}")
    
    return stats['tls'] + stats['http'] + stats['tcp']


def process_folder(input_folder, output_folder):
    """
    处理文件夹中的所有PCAP文件
    保持原有目录结构
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # 支持的PCAP扩展名
    pcap_extensions = ['.pcap', '.pcapng', '.cap']
    
    # 查找所有PCAP文件
    pcap_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if any(file.lower().endswith(ext) for ext in pcap_extensions):
                pcap_files.append(os.path.join(root, file))
    
    logging.info(f"找到 {len(pcap_files)} 个PCAP文件")
    
    total_flows = 0
    
    for pcap_file in pcap_files:
        try:
            # 传递 input_folder 参数，保持目录结构
            flow_count = extract_flows_from_pcap(pcap_file, output_folder, input_folder)
            total_flows += flow_count
        except Exception as e:
            logging.error(f"处理文件 {pcap_file} 时出错: {e}")
    
    logging.info(f"处理完成! 共提取 {total_flows} 个流簇")


def main():
    """主函数：解析参数并启动处理"""
    parser = argparse.ArgumentParser(description="PCAP流簇提取工具（优化版）")
    parser.add_argument("--input", "-i", type=str, default="./webshell_flows",
                        help="输入文件夹路径 (默认: ./webshell_flows)")
    parser.add_argument("--output", "-o", type=str, default="./webshell_flows_flows",
                        help="输出文件夹路径 (默认: ./webshell_flows_flows)")
    args = parser.parse_args()
    
    # 启动处理
    process_folder(args.input, args.output)


if __name__ == "__main__":
    main()