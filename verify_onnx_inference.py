import json
import joblib
import onnxruntime as rt
import numpy as np
from sklearn.model_selection import train_test_split
from typing import cast

# --- 0. 从train_multitask_model.py导入必要函数 ---
from train_realistic_end_to_end_networks import generate_realistic_network_data

# --- 1. 生成并准备完整数据集 ---
print("🔄 Generating and preparing data...")
X_raw, y_binary, _ = generate_realistic_network_data(n_samples=1000) # 生成一小批数据用于寻找

# 加载与模型匹配的scaler
scaler = joblib.load('multitask_scaler.pkl')
X_scaled = scaler.transform(X_raw)

# --- 2. 加载ONNX模型 ---
sess = rt.InferenceSession('multitask_model.onnx')
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
print("✅ Files loaded successfully.")

# --- 3. 寻找一个被正确识别的正常样本 ---
print("\n🔍 Searching for a correctly identified NORMAL sample...")
found_sample = False
for i in range(len(X_raw)):
    # 只关心正常样本
    if y_binary[i] == 0:
        # 准备单个样本进行推理
        raw_sample = X_raw[i].reshape(1, -1)
        scaled_sample = X_scaled[i].reshape(1, -1)
        
        # 推理
        result_list = sess.run([output_name], {input_name: scaled_sample})
        result_array = cast(np.ndarray, result_list[0]) # 明确告诉linter这是个ndarray
        
        # 解析
        detection_logits = result_array[0, :2]
        detection_probs = np.exp(detection_logits) / np.sum(np.exp(detection_logits))
        is_anomaly = np.argmax(detection_probs) == 0

        # 检查是否是我们想要的：真实标签是正常，模型判断也是正常
        if not is_anomaly:
            print(f"🎉 Found a good normal sample at index {i}!")
            
            # --- 4. 创建最终的JSON测试文件 ---
            final_raw_data = X_raw[i]
            network_data_dict = {
                "wlan0_wireless_quality": float(final_raw_data[0]),
                "wlan0_signal_level": float(final_raw_data[1]),
                "wlan0_noise_level": float(final_raw_data[2]),
                "wlan0_rx_packets": int(final_raw_data[3]),
                "wlan0_tx_packets": int(final_raw_data[4]),
                "wlan0_rx_bytes": int(final_raw_data[5]),
                "wlan0_tx_bytes": int(final_raw_data[6]),
                "gateway_ping_time": float(final_raw_data[7]),
                "dns_resolution_time": float(final_raw_data[8]),
                "memory_usage_percent": float(final_raw_data[9]),
                "cpu_usage_percent": float(final_raw_data[10])
            }
            final_json = {
              "timestamp": "2025-07-11T00:00:00Z",
              "device_id": "verified_normal_device",
              "network_data": network_data_dict
            }
            
            output_filename = "final_verified_normal_input.json"
            with open(output_filename, 'w') as f:
                json.dump(final_json, f, indent=2)

            print(f"✅ Final verified test file created: {output_filename}")
            print("Sample values:")
            print(json.dumps(network_data_dict, indent=2))
            
            found_sample = True
            break

if not found_sample:
    print("\n❌ Could not find a correctly identified normal sample in the dataset.")
    print("This indicates a more severe issue with the model's ability to generalize.") 