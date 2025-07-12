import joblib

# 加载之前保存的scaler
scaler = joblib.load('multitask_scaler.pkl')

# 提取均值和缩放（标准差）参数
means = scaler.mean_
scales = scaler.scale_

print("C++ aRRAY dEFINITIONS:")
print("="*30)

# 打印C++格式的均值数组
print("const std::vector<float> SCALER_MEANS = {")
for mean in means:
    print(f"    {mean:.8f}f,")
print("};")

print("\n// --- \n")

# 打印C++格式的缩放（标准差）数组
print("const std::vector<float> SCALER_SCALES = {")
for scale in scales:
    print(f"    {scale:.8f}f,")
print("};") 