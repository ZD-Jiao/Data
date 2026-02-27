import os
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt

def print_hdf5_structure(name, obj):
    """
    回调函数：用于递归遍历并打印 HDF5 文件的内部结构
    """
    # 打印当前的路径名
    indent = name.count('/') * '  '
    if isinstance(obj, h5py.Group):
        print(f"{indent}📁 组 (Group): {name}")
        # 打印组的属性 (Metadata)
        if obj.attrs:
            print(f"{indent}   - 属性 (Attributes):")
            for key, val in obj.attrs.items():
                print(f"{indent}     * {key}: {val}")
    elif isinstance(obj, h5py.Dataset):
        print(f"{indent}📊 数据集 (Dataset): {name}")
        print(f"{indent}   - 形状 (Shape): {obj.shape}")
        print(f"{indent}   - 类型 (Dtype): {obj.dtype}")
        # 如果是复合数据类型 (像 emg2pose 中的 timeseries)，打印内部字段
        if obj.dtype.names:
            print(f"{indent}   - 内部字段 (Fields): {obj.dtype.names}")

def main():
    # 1. 确定目标路径
    data_dir = "./emg2pose_github/emg2pose_dataset_mini"
    
    # 查找目录下的所有 .hdf5 文件
    search_pattern = os.path.join(data_dir, "*.hdf5")
    hdf5_files = glob.glob(search_pattern)
    
    if not hdf5_files:
        print(f"❌ 在路径 {data_dir} 下没有找到任何 .hdf5 文件！")
        print("请检查路径是否正确，或者数据集是否已经解压。")
        return
    
    # 取第一个文件进行分析
    file_path = hdf5_files[0]
    print(f"✅ 找到文件: {file_path}")
    print("-" * 50)
    print("正在打印 HDF5 文件结构...\n")
    
    # 2. 读取文件并打印结构
    with h5py.File(file_path, 'r') as f:
        # 遍历打印所有键名
        f.visititems(print_hdf5_structure)
        print("-" * 50)
        
        # 3. 提取数据并准备绘图
        # 根据 emg2pose 的数据结构，数据应该在 'emg2pose/timeseries' 中
        if 'emg2pose' in f and 'timeseries' in f['emg2pose']:
            print("正在提取时间序列数据以供绘图...")
            timeseries = f['emg2pose']['timeseries']
            
            # 提取前 2000 个采样点（如果是 2kHz 采样率，这就是 1 秒的数据）
            # 避免一次性加载所有数据导致内存溢出或绘图卡顿
            num_samples = 2000
            
            # 由于 timeseries 是复合数据集，我们可以通过字段名直接提取
            time_data = timeseries['time'][:num_samples]
            emg_data = timeseries['emg'][:num_samples]             # 形状: (num_samples, 16)
            joint_angles = timeseries['joint_angles'][:num_samples] # 形状: (num_samples, 20)
            
            # 为了让 x 轴从 0 开始显示相对时间
            time_relative = time_data - time_data[0]
            
            # 4. 使用 Matplotlib 绘制曲线
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            
            # --- 绘制子图 1: 肌电信号 (EMG) ---
            # 为了画面整洁，我们只画前 3 个通道的 EMG 信号
            for channel in range(3):
                ax1.plot(time_relative, emg_data[:, channel], label=f'EMG 通道 {channel+1}', alpha=0.8)
            ax1.set_title('肌电信号 (EMG) - 前3个通道', fontsize=14)
            ax1.set_ylabel('振幅', fontsize=12)
            ax1.legend(loc='upper right')
            ax1.grid(True, linestyle='--', alpha=0.6)
            
            # --- 绘制子图 2: 关节角度 (Joint Angles) ---
            # 我们只画前 3 个关节角度
            for joint in range(3):
                ax2.plot(time_relative, joint_angles[:, joint], label=f'关节 {joint+1}', alpha=0.8)
            ax2.set_title('真实关节角度 (Ground Truth Pose) - 前3个关节', fontsize=14)
            ax2.set_xlabel('时间 (秒)', fontsize=12)
            ax2.set_ylabel('角度 (弧度)', fontsize=12)
            ax2.legend(loc='upper right')
            ax2.grid(True, linestyle='--', alpha=0.6)
            
            plt.tight_layout()
            plt.show()
            
        else:
            print("❌ 文件中没有找到 'emg2pose/timeseries' 数据结构。")
            print("可能这不是标准的 emg2pose 数据集文件，请参考上面的打印结构自行修改键名。")

if __name__ == "__main__":
    main()
