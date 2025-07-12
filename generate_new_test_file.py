import numpy as np
import json
from train_realistic_end_to_end_networks import generate_realistic_network_data

# --- 1. 生成一批混合数据 ---
# 我们生成一小批数据，然后从中找到第一个正常样本
X, y_binary, _ = generate_realistic_network_data(n_samples=20)

# --- 2. 找到第一个正常样本 ---
normal_sample_index = -1
for i in range(len(y_binary)):
    if y_binary[i] == 0: # 标签0代表正常
        normal_sample_index = i
        break

if normal_sample_index == -1:
    raise RuntimeError("Could not find a normal sample in the generated data. Try increasing n_samples.")

sample_data = X[normal_sample_index]

# --- 3. 构建符合规范的JSON对象 ---
network_data_dict = {
    "wlan0_wireless_quality": float(sample_data[0]),
    "wlan0_signal_level": float(sample_data[1]),
    "wlan0_noise_level": float(sample_data[2]),
    "wlan0_rx_packets": int(sample_data[3]),
    "wlan0_tx_packets": int(sample_data[4]),
    "wlan0_rx_bytes": int(sample_data[5]),
    "wlan0_tx_bytes": int(sample_data[6]),
    "gateway_ping_time": float(sample_data[7]),
    "dns_resolution_time": float(sample_data[8]),
    "memory_usage_percent": float(sample_data[9]),
    "cpu_usage_percent": float(sample_data[10])
}

final_json = {
  "timestamp": "2025-07-11T00:00:00Z",
  "device_id": "new_test_device_01",
  "network_data": network_data_dict
}

# --- 4. 保存到文件 ---
output_filename = "new_normal_input.json"
with open(output_filename, 'w') as f:
    json.dump(final_json, f, indent=2)

print(f"✅ New test file created: {output_filename}")
print("Sample values:")
print(json.dumps(network_data_dict, indent=2)) 