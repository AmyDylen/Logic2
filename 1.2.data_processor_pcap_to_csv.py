import os
import subprocess
import csv
from collections import defaultdict
import argparse
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 常见的服务端口
service_ports = {443, 80, 8443, 993, 995, 5223, 8080, 2222, 21, 22, 23, 25, 53, 110, 143, 3306, 3389}

# 协议优先级：TLS > HTTP > TCP
PROTOCOL_PRIORITY = {
    'tls': 3,
    'http': 2,
    'tcp': 1,
    'unknown': 0
}

def get_pcap_files(directory):
    """获取指定目录中的所有PCAP和PCAPNG文件路径"""
    pcap_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.pcap') or file.endswith('.pcapng'):
                pcap_files.append(os.path.join(root, file))
    return pcap_files

def is_syn_packet(tcp_flags):
    """
    判断是否为SYN包
    """
    try:
        # 检查TCP标志是否为SYN（0x02）
        if isinstance(tcp_flags, str):
            if tcp_flags.strip() == "0x02":
                return True
            # 也可能是十进制表示
            if tcp_flags.strip().isdigit() and int(tcp_flags.strip()) == 2:
                return True
        elif isinstance(tcp_flags, int) and tcp_flags == 2:
            return True
    except:
        pass
    return False

def extract_tls_flows(pcap_file, tshark_path="tshark"):
    """
    从PCAP文件中提取TLS类型的TCP双向流，过滤掉keepalive、重传和控制类型的数据包
    """
    # tshark过滤条件：
    # 1. 只处理TLS数据包
    # 2. 过滤掉TCP keepalive数据包（tcp.len == 0）
    # 3. 过滤掉TCP重传数据包（tcp.analysis.retransmission）
    # 4. 过滤掉TCP控制类型数据包（只保留PSH或ACK标志的数据包）
    filter_expr = "tls and tcp.len > 0 and not tcp.analysis.retransmission and (tcp.flags & 0x08 != 0 or tcp.flags & 0x10 != 0)"
    
    command = [
        tshark_path,
        "-r", pcap_file,
        "-Y", filter_expr,
        "-T", "fields",
        "-e", "frame.time_relative",
        "-e", "ip.src",
        "-e", "ip.dst",
        "-e", "tcp.srcport",
        "-e", "tcp.dstport",
        "-e", "tcp.len",
        "-e", "frame.protocols",
        # TLS相关字段
        '-e', 'tls.record.length',
        '-e', 'tls.record.content_type',
        '-e', 'tls.record.opaque_type',  # 添加TLS 1.3的opaque_type字段
        '-e', 'tls.handshake.type',  # 添加TLS握手类型字段
        '-e', 'tls.handshake.version',  # 添加TLS版本字段
        '-e', 'tls.record.version',  # 添加TLS记录版本字段
        # TCP标志字段，用于识别SYN包
        '-e', 'tcp.flags',
        "-E", "separator=|",
        "-E", "occurrence=f"
    ]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output = result.stdout
        return output
    except subprocess.CalledProcessError as e:
        logging.error(f"An error occurred while running tshark on {pcap_file}: {e}")
        logging.error(f"stderr output: {e.stderr}")
        return ""

def extract_http_flows(pcap_file, tshark_path="tshark"):
    """
    从PCAP文件中提取HTTP类型的TCP双向流，过滤掉keepalive、重传和控制类型的数据包
    """
    # tshark过滤条件：
    # 1. 只处理HTTP数据包
    # 2. 过滤掉TCP keepalive数据包（tcp.len == 0）
    # 3. 过滤掉TCP重传数据包（tcp.analysis.retransmission）
    # 4. 过滤掉TCP控制类型数据包（只保留PSH或ACK标志的数据包）
    filter_expr = "http and tcp.len > 0 and not tcp.analysis.retransmission and (tcp.flags & 0x08 != 0 or tcp.flags & 0x10 != 0)"
    
    command = [
        tshark_path,
        "-r", pcap_file,
        "-Y", filter_expr,
        "-T", "fields",
        "-e", "frame.time_relative",
        "-e", "ip.src",
        "-e", "ip.dst",
        "-e", "tcp.srcport",
        "-e", "tcp.dstport",
        "-e", "tcp.len",
        "-e", "frame.protocols",
        # HTTP相关字段
        '-e', 'http.request.method',  # HTTP请求方法
        '-e', 'http.response.code',  # HTTP响应代码
        '-e', 'http.content_length',  # HTTP内容长度
        # TCP标志字段，用于识别SYN包
        '-e', 'tcp.flags',
        "-E", "separator=|",
        "-E", "occurrence=f"
    ]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output = result.stdout
        return output
    except subprocess.CalledProcessError as e:
        logging.error(f"An error occurred while running tshark on {pcap_file}: {e}")
        logging.error(f"stderr output: {e.stderr}")
        return ""

def extract_tcp_flows(pcap_file, tshark_path="tshark"):
    """
    从PCAP文件中提取所有TCP双向流，过滤掉keepalive、重传和控制类型的数据包
    """
    # tshark过滤条件：
    # 1. 只处理TCP数据包
    # 2. 过滤掉TCP keepalive数据包（tcp.len == 0）
    # 3. 过滤掉TCP重传数据包（tcp.analysis.retransmission）
    # 4. 过滤掉TCP控制类型数据包（只保留PSH或ACK标志的数据包）
    filter_expr = "tcp and tcp.len > 0 and not tcp.analysis.retransmission and (tcp.flags & 0x08 != 0 or tcp.flags & 0x10 != 0)"
    
    command = [
        tshark_path,
        "-r", pcap_file,
        "-Y", filter_expr,
        "-T", "fields",
        "-e", "frame.time_relative",
        "-e", "ip.src",
        "-e", "ip.dst",
        "-e", "tcp.srcport",
        "-e", "tcp.dstport",
        "-e", "tcp.len",
        "-e", "frame.protocols",
        # TCP标志字段，用于识别SYN包
        '-e', 'tcp.flags',
        "-E", "separator=|",
        "-E", "occurrence=f"
    ]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output = result.stdout
        return output
    except subprocess.CalledProcessError as e:
        logging.error(f"An error occurred while running tshark on {pcap_file}: {e}")
        logging.error(f"stderr output: {e.stderr}")
        return ""

# classify_protocol 函数不再需要，因为我们已经在 process_pcap_file 中直接确定流的协议类型

def process_tls_flow(flow_packets, session_id, server_info):
    """
    处理TLS类型的流
    只要流中包含TLS数据包，整个流就被视为TLS流
    """
    processed_packets = []
    
    # 为整个流确定统一的正向（客户端）IP
    flow_positive_direction = None
    if flow_packets:
        first_packet = flow_packets[0]
        src_ip = first_packet['src_ip']
        dst_ip = first_packet['dst_ip']
        ip_pair = tuple(sorted([src_ip, dst_ip]))
        
        if ip_pair in server_info:
            server_ip, server_port = server_info[ip_pair]
            # 客户端是与服务器IP不同的IP
            if src_ip != server_ip:
                flow_positive_direction = src_ip
            else:
                flow_positive_direction = dst_ip
        else:
            # 启发式：若目标端口是常见服务端口，则源 IP 是客户端
            dst_port = first_packet['dst_port']
            src_port = first_packet['src_port']
            if dst_port in service_ports:
                flow_positive_direction = src_ip
            elif src_port in service_ports:
                flow_positive_direction = dst_ip
            else:
                # 非标准端口：假设源 IP 是客户端
                flow_positive_direction = src_ip
    
    for packet in flow_packets:
        # 只处理TLS Application Data类型的数据包
        is_tls_application_data = False
        tls_content_type = packet.get('tls_content_type')
        tls_opaque_type = packet.get('tls_opaque_type')
        tls_handshake_type = packet.get('tls_handshake_type')
        
        # 检查是否是TLS Application Data
        if tls_content_type == 23 or tls_opaque_type == 23:
            is_tls_application_data = True
        
        # 检查是否同时包含密钥协商字段
        contains_handshake = tls_handshake_type is not None and tls_handshake_type != 0
        
        # 只处理只包含application数据的数据包
        if is_tls_application_data and not contains_handshake:
            # 计算调整后的长度
            length = packet.get('length', 0)
            tls_length = packet.get('tls_length', 0)
            
            if tls_length < length - 5:
                adjusted_length = length - 5
            elif tls_length > length:
                adjusted_length = tls_length
            else:
                adjusted_length = tls_length
            
            # 确定客户端和服务器
            src_ip = packet['src_ip']
            dst_ip = packet['dst_ip']
            
            # 使用流级别的正向（客户端）IP
            positive_direction = flow_positive_direction
            
            # 确定长度的符号
            final_length = adjusted_length if src_ip == positive_direction else -adjusted_length
            
            # 确定TLS版本
            tls_version = None
            
            # 优先检查是否是TLS 1.3（存在opaque_type字段）
            if tls_opaque_type is not None:
                tls_version = "tls1.3"
            # 否则，从版本字段提取具体版本
            else:
                # 尝试从tls.record.version提取
                tls_record_version = packet.get('tls_record_version')
                if tls_record_version:
                    if tls_record_version == "0x0304":
                        tls_version = "tls1.3"
                    elif tls_record_version == "0x0303":
                        tls_version = "tls1.2"
                    elif tls_record_version == "0x0302":
                        tls_version = "tls1.1"
                    elif tls_record_version == "0x0301":
                        tls_version = "tls1.0"
                    elif tls_record_version == "0x0300":
                        tls_version = "ssl3.0"
                
                # 如果没有获取到，尝试从tls.handshake.version提取
                if tls_version is None:
                    tls_handshake_version = packet.get('tls_handshake_version')
                    if tls_handshake_version:
                        if tls_handshake_version == "0x0304":
                            tls_version = "tls1.3"
                        elif tls_handshake_version == "0x0303":
                            tls_version = "tls1.2"
                        elif tls_handshake_version == "0x0302":
                            tls_version = "tls1.1"
                        elif tls_handshake_version == "0x0301":
                            tls_version = "tls1.0"
                        elif tls_handshake_version == "0x0300":
                            tls_version = "ssl3.0"
                
                # 如果仍然没有获取到，默认为tls1.2
                if tls_version is None:
                    tls_version = "tls1.2"
            
            # 存储处理后的数据
            processed_packets.append({
                'session_id': session_id,
                'length': final_length,
                'timestamp': packet['timestamp'],
                'src_ip': src_ip,
                'dst_ip': dst_ip,
                'src_port': packet['src_port'],
                'dst_port': packet['dst_port'],
                'protocols': packet['protocols'],
                'type': 'packet',
                'tls_version': tls_version
            })
    
    return processed_packets

def process_http_flow(flow_packets, session_id, server_info):
    """
    处理HTTP类型的流
    只要流中包含HTTP数据包，整个流就被视为HTTP流
    """
    processed_packets = []
    
    # 为整个流确定统一的正向（客户端）IP
    flow_positive_direction = None
    if flow_packets:
        first_packet = flow_packets[0]
        src_ip = first_packet['src_ip']
        dst_ip = first_packet['dst_ip']
        ip_pair = tuple(sorted([src_ip, dst_ip]))
        
        if ip_pair in server_info:
            server_ip, server_port = server_info[ip_pair]
            # 客户端是与服务器IP不同的IP
            if src_ip != server_ip:
                flow_positive_direction = src_ip
            else:
                flow_positive_direction = dst_ip
        else:
            # 启发式：若目标端口是常见服务端口，则源 IP 是客户端
            dst_port = first_packet['dst_port']
            src_port = first_packet['src_port']
            if dst_port in service_ports:
                flow_positive_direction = src_ip
            elif src_port in service_ports:
                flow_positive_direction = dst_ip
            else:
                # 非标准端口：假设源 IP 是客户端
                flow_positive_direction = src_ip
    
    for packet in flow_packets:
        # 只处理包含HTTP信息的数据包
        http_method = packet.get('http_method')
        http_response_code = packet.get('http_response_code')
        
        if http_method is not None or http_response_code is not None:
            # 确定客户端和服务器
            src_ip = packet['src_ip']
            dst_ip = packet['dst_ip']
            
            # 使用流级别的正向（客户端）IP
            positive_direction = flow_positive_direction
            
            # 计算长度
            length = 0
            
            # 对于HTTP响应报文，优先使用Content-Length
            if http_response_code is not None:
                http_content_length = packet.get('http_content_length')
                if http_content_length:
                    try:
                        # 尝试将Content-Length转换为整数
                        length = int(http_content_length)
                    except (ValueError, TypeError):
                        # 如果Content-Length格式不正确，使用TCP载荷长度
                        length = packet.get('length', 0)
                else:
                    # 如果没有Content-Length，使用TCP载荷长度
                    length = packet.get('length', 0)
                
                # 确保HTTP响应长度不为0，最小为1
                if length <= 0:
                    length = 1
            else:
                # 对于HTTP请求报文，根据请求类型计算长度
                http_method = packet.get('http_method', '').lower()
                
                # 这里我们假设请求字段的长度可以从HTTP头部或TCP载荷中获取
                # 由于我们没有直接的HTTP请求字段长度，我们使用TCP载荷长度
                # 并根据请求类型进行调整
                tcp_length = packet.get('length', 0)
                
                if http_method == 'post':
                    # POST请求，使用TCP载荷长度
                    length = tcp_length
                elif http_method == 'get':
                    # GET请求，使用TCP载荷长度
                    length = tcp_length
                else:
                    # 其他请求类型，使用TCP载荷长度
                    length = tcp_length
                
                # 确保HTTP请求长度不为0，最小为1
                if length <= 0:
                    length = 1
            
            # 确定长度的符号
            final_length = length if src_ip == positive_direction else -length
            
            # 存储处理后的数据
            processed_packets.append({
                'session_id': session_id,
                'length': final_length,
                'timestamp': packet['timestamp'],
                'src_ip': src_ip,
                'dst_ip': dst_ip,
                'src_port': packet['src_port'],
                'dst_port': packet['dst_port'],
                'protocols': packet['protocols'],
                'type': 'packet'
            })
    
    return processed_packets

def process_tcp_flow(flow_packets, session_id, server_info):
    """
    处理纯TCP类型的流
    """
    processed_packets = []
    
    # 为整个流确定统一的正向（客户端）IP
    flow_positive_direction = None
    if flow_packets:
        first_packet = flow_packets[0]
        src_ip = first_packet['src_ip']
        dst_ip = first_packet['dst_ip']
        ip_pair = tuple(sorted([src_ip, dst_ip]))
        
        if ip_pair in server_info:
            server_ip, server_port = server_info[ip_pair]
            # 客户端是与服务器IP不同的IP
            if src_ip != server_ip:
                flow_positive_direction = src_ip
            else:
                flow_positive_direction = dst_ip
        else:
            # 启发式：若目标端口是常见服务端口，则源 IP 是客户端
            dst_port = first_packet['dst_port']
            src_port = first_packet['src_port']
            if dst_port in service_ports:
                flow_positive_direction = src_ip
            elif src_port in service_ports:
                flow_positive_direction = dst_ip
            else:
                # 非标准端口：假设源 IP 是客户端
                flow_positive_direction = src_ip
    
    for packet in flow_packets:
        # 计算调整后的长度（使用tcp.len）
        length = packet.get('length', 0)
        
        # 确定客户端和服务器
        src_ip = packet['src_ip']
        dst_ip = packet['dst_ip']
        
        # 使用流级别的正向（客户端）IP
        positive_direction = flow_positive_direction
        
        # 确定长度的符号
        final_length = length if src_ip == positive_direction else -length
        
        # 存储处理后的数据
        processed_packets.append({
            'session_id': session_id,
            'length': final_length,
            'timestamp': packet['timestamp'],
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'src_port': packet['src_port'],
            'dst_port': packet['dst_port'],
            'protocols': packet['protocols'],
            'type': 'packet'
        })
    
    return processed_packets

def merge_blocks(packets):
    """
    合并时间间隔小于1秒的同方向数据包
    """
    if not packets:
        return []
    
    # 按时间戳排序
    sorted_packets = sorted(packets, key=lambda x: x['timestamp'])
    
    merged_blocks = []
    current_block = sorted_packets[0].copy()
    
    for packet in sorted_packets[1:]:
        # 检查是否同方向
        same_direction = (current_block['src_ip'] == packet['src_ip'])
        # 检查时间间隔
        time_gap = packet['timestamp'] - current_block['timestamp']
        
        if same_direction and time_gap <= 1000000:  # 1秒 = 1,000,000微秒
            # 合并：更新时间戳为当前包时间，累加长度
            current_block['length'] += packet['length']
            current_block['timestamp'] = packet['timestamp']
        else:
            merged_blocks.append(current_block)
            current_block = packet.copy()
    
    merged_blocks.append(current_block)
    return merged_blocks

def create_flow_key(src_ip, dst_ip, src_port, dst_port):
    """
    创建标准化的流键，确保同一对IP和端口无论方向如何都被视为同一个流族
    按照三元组（源IP，目的IP，目的端口）进行划分
    """
    # 定义常见的标准端口
    standard_ports = {443, 80, 8443, 993, 995, 5223, 8080, 2222}
    
    # 确定标准端口
    try:
        # 尝试将端口转换为整数进行比较
        src_port_int = int(src_port)
        dst_port_int = int(dst_port)
        
        # 对于双向流，确定标准端口
        if src_port_int in standard_ports:
            standard_port = src_port_int
        elif dst_port_int in standard_ports:
            standard_port = dst_port_int
        else:
            # 对于非标准端口，使用较小的端口作为标准端口
            standard_port = min(src_port_int, dst_port_int)
        
        # 排序IP，确保同一对IP无论方向如何都被视为同一个流族
        sorted_ips = sorted([src_ip, dst_ip])
        
        return (sorted_ips[0], sorted_ips[1], standard_port)
    except (ValueError, TypeError):
        # 如果端口转换失败，使用原始的端口值
        # 排序IP，确保同一对IP无论方向如何都被视为同一个流族
        sorted_ips = sorted([src_ip, dst_ip])
        
        # 尝试确定标准端口
        try:
            # 检查是否是标准端口
            if str(src_port) in standard_ports:
                standard_port = src_port
            elif str(dst_port) in standard_ports:
                standard_port = dst_port
            else:
                # 对于非标准端口，使用较小的端口作为标准端口
                if str(src_port) < str(dst_port):
                    standard_port = src_port
                else:
                    standard_port = dst_port
        except:
            # 如果所有尝试都失败，使用原始的dst_port作为标准端口
            standard_port = dst_port
        
        return (sorted_ips[0], sorted_ips[1], standard_port)

def parse_tshark_output(output, is_tls=False, is_http=False):
    """
    解析tshark输出，构建双向流
    """
    session_data = defaultdict(list)
    server_info = {}
    
    for line in output.strip().split('\n'):
        if not line:
            continue
        
        fields = line.split('|')
        if len(fields) < 7:
            continue
        
        try:
            # 解析基本字段
            relative_timestamp = int(float(fields[0])*1000000)  # 转为微秒整数
            src_ip = fields[1]
            dst_ip = fields[2]
            src_port = fields[3]
            dst_port = fields[4]
            tcp_len = fields[5]
            protocols = fields[6]
            
            # 初始化变量
            tls_content_type = None
            tls_opaque_type = None
            tls_handshake_type = None
            tls_length = 0
            tls_handshake_version = None
            tls_record_version = None
            http_method = None
            http_response_code = None
            http_content_length = None
            
            # 提取TLS相关字段
            if is_tls and len(fields) > 7:
                if fields[7]:
                    tls_length = int(fields[7]) if fields[7].isdigit() else 0
                if len(fields) > 8 and fields[8]:
                    tls_content_type = int(fields[8]) if fields[8].isdigit() else None
                if len(fields) > 9 and fields[9]:
                    tls_opaque_type = int(fields[9]) if fields[9].isdigit() else None
                if len(fields) > 10 and fields[10]:
                    tls_handshake_type = int(fields[10]) if fields[10].isdigit() else None
                if len(fields) > 11 and fields[11]:
                    tls_handshake_version = fields[11]
                if len(fields) > 12 and fields[12]:
                    tls_record_version = fields[12]
            
            # 提取HTTP相关字段
            if is_http and len(fields) > 7:
                if len(fields) > 7 and fields[7]:
                    http_method = fields[7]
                if len(fields) > 8 and fields[8]:
                    http_response_code = fields[8]
                if len(fields) > 9 and fields[9]:
                    http_content_length = fields[9]
                http_chunk_size = None
            
            # 提取TCP标志字段，用于识别SYN包
            tcp_flags = None
            if is_tls:
                # TLS流的字段顺序：frame.time_relative, ip.src, ip.dst, tcp.srcport, tcp.dstport, tcp.len, frame.protocols, tls.record.length, tls.record.content_type, tls.record.opaque_type, tls.handshake.type, tls.handshake.version, tls.record.version, tcp.flags
                if len(fields) > 13 and fields[13]:
                    tcp_flags = fields[13]
                elif len(fields) > 12 and fields[12]:
                    tcp_flags = fields[12]
                elif len(fields) > 11 and fields[11]:
                    tcp_flags = fields[11]
            elif is_http:
                # HTTP流的字段顺序：frame.time_relative, ip.src, ip.dst, tcp.srcport, tcp.dstport, tcp.len, frame.protocols, http.request.method, http.response.code, http.content_length, http.chunk.size, tcp.flags
                if len(fields) > 11 and fields[11]:
                    tcp_flags = fields[11]
                elif len(fields) > 10 and fields[10]:
                    tcp_flags = fields[10]
                elif len(fields) > 9 and fields[9]:
                    tcp_flags = fields[9]
                elif len(fields) > 8 and fields[8]:
                    tcp_flags = fields[8]
            else:  # TCP
                # TCP流的字段顺序：frame.time_relative, ip.src, ip.dst, tcp.srcport, tcp.dstport, tcp.len, frame.protocols, tcp.flags
                if len(fields) > 7 and fields[7]:
                    tcp_flags = fields[7]
                elif len(fields) > 6 and fields[6]:
                    tcp_flags = fields[6]
            
            # 处理SYN包，确定服务器信息
            is_syn = is_syn_packet(tcp_flags)
            if is_syn:
                # SYN包的目的IP和端口是服务器
                ip_pair = tuple(sorted([src_ip, dst_ip]))
                server_info[ip_pair] = (dst_ip, dst_port)
            
            # 计算长度
            length = int(tcp_len) if tcp_len.isdigit() else 0
            
            # 生成双向流键
            flow_key = frozenset([(src_ip, dst_ip, src_port, dst_port), (dst_ip, src_ip, dst_port, src_port)])
            
            # 存储数据包信息
            packet_info = {
                'timestamp': relative_timestamp,
                'src_ip': src_ip,
                'dst_ip': dst_ip,
                'src_port': src_port,
                'dst_port': dst_port,
                'length': length,
                'protocols': protocols
            }
            
            # 添加TLS相关字段
            if is_tls:
                packet_info.update({
                    'tls_content_type': tls_content_type,
                    'tls_opaque_type': tls_opaque_type,
                    'tls_handshake_type': tls_handshake_type,
                    'tls_length': tls_length,
                    'tls_handshake_version': tls_handshake_version,
                    'tls_record_version': tls_record_version
                })
            
            # 添加HTTP相关字段
            if is_http:
                packet_info.update({
                    'http_method': http_method,
                    'http_response_code': http_response_code,
                    'http_content_length': http_content_length
                })
            
            session_data[flow_key].append(packet_info)
        except Exception as e:
            logging.error(f"Error processing line: {line}")
            logging.error(f"Error: {e}")
            continue
    
    return session_data, server_info

def process_pcap_file(pcap_file, output_folder, tshark_path="tshark", input_base_folder=None):
    """
    处理单个PCAP文件
    新流程：优先处理TLS流，然后处理HTTP流，最后处理剩余的TCP流
    
    Args:
        pcap_file: PCAP文件完整路径
        output_folder: 输出文件夹路径
        tshark_path: tshark可执行文件路径
        input_base_folder: 输入基础文件夹路径，用于保留子目录结构
    """
    logging.info(f"Processing PCAP file: {pcap_file}")
    
    # 提取文件名（不含扩展名）
    file_name = os.path.splitext(os.path.basename(pcap_file))[0]
    
    # 计算相对路径，保留子目录结构
    if input_base_folder:
        rel_path = os.path.relpath(os.path.dirname(pcap_file), input_base_folder)
        if rel_path == '.':
            rel_path = ''
    else:
        rel_path = ''
    
    # 构建输出子文件夹路径
    output_subfolder = os.path.join(output_folder, rel_path)
    os.makedirs(output_subfolder, exist_ok=True)
    
    # 构建基础文件名：子文件夹名_文件名
    if rel_path:
        # 将路径分隔符替换为下划线
        folder_prefix = rel_path.replace(os.sep, '_')
        base_name = f"{folder_prefix}_{file_name}"
    else:
        base_name = file_name
    
    # 首先提取所有TCP流，用于获取完整的服务器信息（包括SYN包）
    logging.info(f"Extracting all TCP flows to get server information")
    tcp_output = extract_tcp_flows(pcap_file, tshark_path)
    _, all_server_info = parse_tshark_output(tcp_output)
    
    # 存储已处理的流的四元组信息
    processed_flows = set()
    
    # 存储处理后的会话
    processed_sessions = []
    session_id = 1
    
    # 1. 处理TLS流
    logging.info(f"Processing TLS flows in {pcap_file}")
    tls_output = extract_tls_flows(pcap_file, tshark_path)
    tls_session_data, _ = parse_tshark_output(tls_output, is_tls=True)
    
    for flow_key, packets in tls_session_data.items():
        if not packets:
            continue
        
        # 标记为已处理
        processed_flows.add(flow_key)
        
        # 确定TLS版本
        # 收集所有可能的TLS版本
        tls_versions = []
        
        # 遍历所有数据包，收集所有可能的TLS版本
        for packet in packets:
            # 检查TLS 1.3（存在opaque_type字段）
            if packet.get('tls_opaque_type') is not None:
                tls_versions.append("tls1.3")
            
            # 从tls.record.version提取版本
            tls_record_version = packet.get('tls_record_version')
            if tls_record_version:
                if tls_record_version == "0x0304":
                    tls_versions.append("tls1.3")
                elif tls_record_version == "0x0303":
                    tls_versions.append("tls1.2")
                elif tls_record_version == "0x0302":
                    tls_versions.append("tls1.1")
                elif tls_record_version == "0x0301":
                    tls_versions.append("tls1.0")
                elif tls_record_version == "0x0300":
                    tls_versions.append("ssl3.0")
            
            # 从tls.handshake.version提取版本
            tls_handshake_version = packet.get('tls_handshake_version')
            if tls_handshake_version:
                if tls_handshake_version == "0x0304":
                    tls_versions.append("tls1.3")
                elif tls_handshake_version == "0x0303":
                    tls_versions.append("tls1.2")
                elif tls_handshake_version == "0x0302":
                    tls_versions.append("tls1.1")
                elif tls_handshake_version == "0x0301":
                    tls_versions.append("tls1.0")
                elif tls_handshake_version == "0x0300":
                    tls_versions.append("ssl3.0")
        
        # 确定使用的TLS版本
        tls_version = None
        
        if tls_versions:
            # 统计每个版本的出现次数
            from collections import Counter
            version_counts = Counter(tls_versions)
            
            # 按出现次数排序
            sorted_versions = sorted(version_counts.items(), key=lambda x: x[1], reverse=True)
            
            # 检查是否有多个版本出现次数相同
            if len(sorted_versions) > 1 and sorted_versions[0][1] == sorted_versions[1][1]:
                # 如果出现次数相同，优先选择较高的版本
                version_priority = {"tls1.3": 4, "tls1.2": 3, "tls1.1": 2, "tls1.0": 1, "ssl3.0": 0}
                highest_priority = -1
                for version, _ in sorted_versions:
                    if version_priority.get(version, -1) > highest_priority:
                        highest_priority = version_priority.get(version, -1)
                        tls_version = version
            else:
                # 使用出现次数最多的版本
                tls_version = sorted_versions[0][0]
        
        # 如果仍然没有获取到，默认为tls1.2
        if tls_version is None:
            tls_version = "tls1.2"
        
        # 处理TLS流，使用统一的服务器信息
        protocol = "tls"
        processed_packets = process_tls_flow(packets, session_id, all_server_info)
        
        if not processed_packets:
            continue
        
        # 合并时间间隔小于1秒的同方向数据包
        merged_blocks = merge_blocks(processed_packets)
        
        if not merged_blocks:
            continue
        
        # 计算会话持续时间
        session_duration = merged_blocks[-1]['timestamp'] - merged_blocks[0]['timestamp']
        packet_count = len(processed_packets)
        block_count = len(merged_blocks)
        
        # 存储处理后的会话
        for block in merged_blocks:
            processed_sessions.append({
                'session_id': session_id,
                'length': block['length'],
                'timestamp': block['timestamp'],
                'session_duration': session_duration,
                'src_ip': block['src_ip'],
                'dst_ip': block['dst_ip'],
                'src_port': block['src_port'],
                'dst_port': block['dst_port'],
                'protocols': block['protocols'],
                'type': 'block',
                'packet_count': packet_count,
                'block_count': block_count,
                'protocol': protocol,
                'tls_version': tls_version
            })
        
        session_id += 1
    
    # 2. 处理HTTP流
    logging.info(f"Processing HTTP flows in {pcap_file}")
    http_output = extract_http_flows(pcap_file, tshark_path)
    http_session_data, _ = parse_tshark_output(http_output, is_http=True)
    
    for flow_key, packets in http_session_data.items():
        if not packets:
            continue
        
        # 检查是否已经处理过（可能是TLS流）
        if flow_key in processed_flows:
            continue
        
        # 标记为已处理
        processed_flows.add(flow_key)
        
        # 处理HTTP流，使用统一的服务器信息
        protocol = "http"
        processed_packets = process_http_flow(packets, session_id, all_server_info)
        
        if not processed_packets:
            continue
        
        # 合并时间间隔小于1秒的同方向数据包
        merged_blocks = merge_blocks(processed_packets)
        
        if not merged_blocks:
            continue
        
        # 计算会话持续时间
        session_duration = merged_blocks[-1]['timestamp'] - merged_blocks[0]['timestamp']
        packet_count = len(processed_packets)
        block_count = len(merged_blocks)
        
        # 存储处理后的会话
        for block in merged_blocks:
            processed_sessions.append({
                'session_id': session_id,
                'length': block['length'],
                'timestamp': block['timestamp'],
                'session_duration': session_duration,
                'src_ip': block['src_ip'],
                'dst_ip': block['dst_ip'],
                'src_port': block['src_port'],
                'dst_port': block['dst_port'],
                'protocols': block['protocols'],
                'type': 'block',
                'packet_count': packet_count,
                'block_count': block_count,
                'protocol': protocol,
                'tls_version': ''
            })
        
        session_id += 1
    
    # 3. 处理剩余的TCP流
    logging.info(f"Processing remaining TCP flows in {pcap_file}")
    tcp_session_data, _ = parse_tshark_output(tcp_output)
    
    for flow_key, packets in tcp_session_data.items():
        if not packets:
            continue
        
        # 检查是否已经处理过（可能是TLS或HTTP流）
        if flow_key in processed_flows:
            continue
        
        # 标记为已处理
        processed_flows.add(flow_key)
        
        # 处理TCP流，使用统一的服务器信息
        protocol = "tcp"
        processed_packets = process_tcp_flow(packets, session_id, all_server_info)
        
        if not processed_packets:
            continue
        
        # 合并时间间隔小于1秒的同方向数据包
        merged_blocks = merge_blocks(processed_packets)
        
        if not merged_blocks:
            continue
        
        # 计算会话持续时间
        session_duration = merged_blocks[-1]['timestamp'] - merged_blocks[0]['timestamp']
        packet_count = len(processed_packets)
        block_count = len(merged_blocks)
        
        # 存储处理后的会话
        for block in merged_blocks:
            processed_sessions.append({
                'session_id': session_id,
                'length': block['length'],
                'timestamp': block['timestamp'],
                'session_duration': session_duration,
                'src_ip': block['src_ip'],
                'dst_ip': block['dst_ip'],
                'src_port': block['src_port'],
                'dst_port': block['dst_port'],
                'protocols': block['protocols'],
                'type': 'block',
                'packet_count': packet_count,
                'block_count': block_count,
                'protocol': protocol,
                'tls_version': ''
            })
        
        session_id += 1
    
    # 按协议类型分组生成CSV文件
    if processed_sessions:
        # 按协议类型分组
        protocol_groups = defaultdict(list)
        for session in processed_sessions:
            protocol = session.get('protocol', 'unknown')
            protocol_groups[protocol].append(session)
        
        generated_files = []
        
        # 为每种协议生成单独的CSV文件
        for protocol, sessions in protocol_groups.items():
            # 确定文件后缀
            if protocol == 'tls':
                # TLS协议需要进一步按版本分组
                tls_version_groups = defaultdict(list)
                for session in sessions:
                    tls_version = session.get('tls_version', '')
                    if tls_version:
                        tls_version_groups[tls_version].append(session)
                    else:
                        tls_version_groups['tls1.2'].append(session)  # 默认版本
                
                # 为每个TLS版本生成文件
                for tls_version, version_sessions in tls_version_groups.items():
                    output_csv_file = os.path.join(output_subfolder, f"{base_name}_{tls_version}.csv")
                    
                    with open(output_csv_file, mode='w', newline='', encoding='utf-8') as file:
                        writer = csv.writer(file)
                        writer.writerow([
                            "Session ID", "Length", "Timestamp", "Session Duration",
                            "Source IP", "Destination IP", "Source Port", "Destination Port",
                            "Protocols", "Type", "Packet Count", "Block Count", "Protocol", "TLS Version"
                        ])
                        
                        for session in version_sessions:
                            writer.writerow([
                                session['session_id'],
                                session['length'],
                                session['timestamp'],
                                session['session_duration'],
                                session['src_ip'],
                                session['dst_ip'],
                                session['src_port'],
                                session['dst_port'],
                                session['protocols'],
                                session['type'],
                                session['packet_count'],
                                session['block_count'],
                                session['protocol'],
                                session.get('tls_version', '')
                            ])
                    
                    logging.info(f"Generated CSV file: {output_csv_file} ({len(version_sessions)} sessions)")
                    generated_files.append(output_csv_file)
            
            elif protocol == 'http':
                output_csv_file = os.path.join(output_subfolder, f"{base_name}_http.csv")
                
                with open(output_csv_file, mode='w', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        "Session ID", "Length", "Timestamp", "Session Duration",
                        "Source IP", "Destination IP", "Source Port", "Destination Port",
                        "Protocols", "Type", "Packet Count", "Block Count", "Protocol", "TLS Version"
                    ])
                    
                    for session in sessions:
                        writer.writerow([
                            session['session_id'],
                            session['length'],
                            session['timestamp'],
                            session['session_duration'],
                            session['src_ip'],
                            session['dst_ip'],
                            session['src_port'],
                            session['dst_port'],
                            session['protocols'],
                            session['type'],
                            session['packet_count'],
                            session['block_count'],
                            session['protocol'],
                            session.get('tls_version', '')
                        ])
                
                logging.info(f"Generated CSV file: {output_csv_file} ({len(sessions)} sessions)")
                generated_files.append(output_csv_file)
            
            elif protocol == 'tcp':
                output_csv_file = os.path.join(output_subfolder, f"{base_name}_tcp.csv")
                
                with open(output_csv_file, mode='w', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        "Session ID", "Length", "Timestamp", "Session Duration",
                        "Source IP", "Destination IP", "Source Port", "Destination Port",
                        "Protocols", "Type", "Packet Count", "Block Count", "Protocol", "TLS Version"
                    ])
                    
                    for session in sessions:
                        writer.writerow([
                            session['session_id'],
                            session['length'],
                            session['timestamp'],
                            session['session_duration'],
                            session['src_ip'],
                            session['dst_ip'],
                            session['src_port'],
                            session['dst_port'],
                            session['protocols'],
                            session['type'],
                            session['packet_count'],
                            session['block_count'],
                            session['protocol'],
                            session.get('tls_version', '')
                        ])
                
                logging.info(f"Generated CSV file: {output_csv_file} ({len(sessions)} sessions)")
                generated_files.append(output_csv_file)
        
        return generated_files
    else:
        logging.info(f"No flows processed for {pcap_file}")
        return None

def process_csv_files(input_folder, output_folder):
    """
    处理文件夹中的所有CSV文件，按流族分组
    """
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 获取所有CSV文件
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    
    for csv_file in csv_files:
        logging.info(f"Processing file: {csv_file}")
        file_path = os.path.join(input_folder, csv_file)
        
        # 读取CSV文件
        try:
            import pandas as pd
            df = pd.read_csv(file_path)
        except Exception as e:
            logging.error(f"Error reading file {csv_file}: {e}")
            continue
        
        # 按流族分组
        flow_groups = defaultdict(list)
        
        for index, row in df.iterrows():
            src_ip = row['Source IP']
            dst_ip = row['Destination IP']
            src_port = row['Source Port']
            dst_port = row['Destination Port']
            flow_key = create_flow_key(src_ip, dst_ip, src_port, dst_port)
            flow_groups[flow_key].append(row)
        
        # 为每个流族生成CSV文件
        base_name = os.path.splitext(csv_file)[0]
        
        for idx, (flow_key, rows) in enumerate(flow_groups.items(), 1):
            # 创建新的DataFrame
            group_df = pd.DataFrame(rows)
            
            # 确定协议类型和TLS版本
            protocol = "unknown"
            tls_version = ""
            
            # 确定协议类型
            if 'Protocol' in group_df.columns:
                protocol_counts = group_df['Protocol'].value_counts()
                if not protocol_counts.empty:
                    protocol = protocol_counts.idxmax()
            
            # 确定TLS版本（如果是TLS协议）
            if protocol == "tls" and 'TLS Version' in group_df.columns:
                version_counts = group_df['TLS Version'].value_counts()
                if not version_counts.empty:
                    tls_version = version_counts.idxmax()
            
            # 生成文件名
            ip1, ip2, port = flow_key
            # 添加协议类型和TLS版本后缀
            if protocol == "tls" and tls_version:
                output_file_name = f"{base_name}_flow_family_{idx}_{ip1}_to_{ip2}_{port}_{tls_version}.csv"
            elif protocol == "http":
                output_file_name = f"{base_name}_flow_family_{idx}_{ip1}_to_{ip2}_{port}_{protocol}.csv"
            elif protocol == "tcp":
                output_file_name = f"{base_name}_flow_family_{idx}_{ip1}_to_{ip2}_{port}_{protocol}.csv"
            else:
                output_file_name = f"{base_name}_flow_family_{idx}_{ip1}_to_{ip2}_{port}.csv"
            output_file_path = os.path.join(output_folder, output_file_name)
            
            # 保存到CSV文件
            try:
                group_df.to_csv(output_file_path, index=False)
                if protocol == "tls" and tls_version:
                    logging.info(f"  Saved flow family {idx}: {ip1} <-> {ip2}:{port} ({tls_version}) ({len(rows)} records)")
                elif protocol == "http":
                    logging.info(f"  Saved flow family {idx}: {ip1} <-> {ip2}:{port} ({protocol}) ({len(rows)} records)")
                elif protocol == "tcp":
                    logging.info(f"  Saved flow family {idx}: {ip1} <-> {ip2}:{port} ({protocol}) ({len(rows)} records)")
                else:
                    logging.info(f"  Saved flow family {idx}: {ip1} <-> {ip2}:{port} ({len(rows)} records)")
            except Exception as e:
                logging.error(f"  Error saving file {output_file_name}: {e}")
                continue
        
        logging.info(f"File {csv_file} processed, generated {len(flow_groups)} flow families")

def process_pcap_files(folder_path, output_folder, tshark_path="tshark"):
    """
    批量处理PCAP文件
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    pcap_files = get_pcap_files(folder_path)
    logging.info(f"Found {len(pcap_files)} PCAP files to process")
    
    # 获取绝对路径作为基础文件夹
    input_base_folder = os.path.abspath(folder_path)
    
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_pcap_file, pcap, output_folder, tshark_path, input_base_folder): pcap for pcap in pcap_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing PCAP files"):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error processing PCAP file: {e}")

def main(input_dir, output_dir, tshark_path="C:\Program Files\Wireshark\tshark.exe", flow_cluster_dir=None):
    """
    主函数
    """
    # 处理PCAP文件
    process_pcap_files(input_dir, output_dir, tshark_path)
    
    # 如果指定了流簇输出文件夹，则生成流簇文件
    if flow_cluster_dir:
        logging.info(f"Generating flow cluster files to {flow_cluster_dir}")
        process_csv_files(output_dir, flow_cluster_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process PCAP files and extract protocol flows",
        epilog="""
Examples:
  # Basic usage - process PCAP files and generate CSV files
  python 1.2.data_processor_pcap_to_csv.py E:\\pcap_input E:\\csv_output

  # Specify custom tshark path
  python 1.2.data_processor_pcap_to_csv.py E:\\pcap_input E:\\csv_output --tshark-path "D:\\Wireshark\\tshark.exe"

  # Generate flow cluster files in addition to CSV files
  python 1.2.data_processor_pcap_to_csv.py E:\\pcap_input E:\\csv_output --flow-cluster-dir E:\\flow_clusters

  # Full example with all options
  python 1.2.data_processor_pcap_to_csv.py \\
      E:\\pcap_input \\
      E:\\csv_output \\
      --tshark-path "C:\\Program Files\\Wireshark\\tshark.exe" \\
      --flow-cluster-dir E:\\flow_clusters
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input_dir", help="Directory containing PCAP files (supports .pcap and .pcapng)")
    parser.add_argument("output_dir", help="Directory to save output CSV files")
    parser.add_argument("--tshark-path", default=r"C:\Program Files\Wireshark\tshark.exe", 
                        help="Path to tshark executable (default: C:\Program Files\Wireshark\tshark.exe)")
    parser.add_argument("--flow-cluster-dir", default=None, 
                        help="Directory to save flow cluster files (optional)")
    args = parser.parse_args()
    
    main(args.input_dir, args.output_dir, args.tshark_path, args.flow_cluster_dir)
