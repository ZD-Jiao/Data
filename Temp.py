import os
import glob
import random
import copy
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from Model_SleepFM import Config, SleepFM_Finetune

# ==========================================
# 数据加载与处理模块
# ==========================================
def extract_windows_from_file(f_path, is_train):
    windows, labels, valid_masks = [], [], []
    with h5py.File(f_path, 'r') as f:
        emg_data = f['emg'][:]
        lbl_data = f['label'][:]
        if isinstance(lbl_data[0], bytes):
            lbl_data = [l.decode('utf-8') for l in lbl_data]
            
        # 记录微调数据集的真实通道有效性
        num_c = emg_data.shape[1]
        v_mask = np.zeros(Config.CHANNELS, dtype=bool)
        v_mask[:min(num_c, Config.CHANNELS)] = True
            
        if num_c < Config.CHANNELS:
            pad = np.zeros((emg_data.shape[0], Config.CHANNELS - num_c), dtype=np.float32)
            emg_data = np.concatenate((emg_data, pad), axis=1)
        else:
            emg_data = emg_data[:, :Config.CHANNELS]
            
        if is_train:
            step = int(Config.WINDOW_SIZE * 0.25)
        else:
            step = Config.WINDOW_SIZE             
            
        for i in range(0, len(emg_data) - Config.WINDOW_SIZE, step):
            win_emg = emg_data[i : i + Config.WINDOW_SIZE].T 
            win_lbl = lbl_data[i + Config.WINDOW_SIZE - 1]
            windows.append(win_emg)
            labels.append(win_lbl)
            valid_masks.append(v_mask)
            
    return windows, labels, valid_masks

class EMGFinetuneDataset(Dataset):
    def __init__(self, windows, labels, valid_masks, label2idx, is_train=False):
        self.windows = np.array(windows, dtype=np.float32)
        self.labels = np.array([label2idx[l] for l in labels], dtype=np.longlong)
        self.valid_masks = np.array(valid_masks, dtype=bool)
        self.is_train = is_train

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = torch.tensor(self.windows[idx])
        y = torch.tensor(self.labels[idx])
        v_mask = torch.tensor(self.valid_masks[idx])
        
        if self.is_train:
            if torch.rand(1).item() < 0.5:
                noise = torch.randn_like(x) * 0.05
                x = x + noise
            if torch.rand(1).item() < 0.5:
                scale = torch.empty(1).uniform_(0.8, 1.2).item()
                x = x * scale
                
            # --- 软化：随机通道屏蔽 (Channel Masking) ---
            # 概率降至 0.2，防止破坏过多特征导致模型坍塌
            if torch.rand(1).item() < 0.2:
                valid_indices = torch.nonzero(v_mask).squeeze()
                if valid_indices.dim() == 0: 
                    valid_indices = valid_indices.unsqueeze(0)
                
                # 最多只随机丢弃 1 个通道的数据
                if len(valid_indices) > 2:
                    num_drop = 1 
                    perm = torch.randperm(len(valid_indices))
                    drop_indices = valid_indices[perm[:num_drop]]
                    x[drop_indices, :] = 0.0
                
        return x, y, v_mask

def prepare_data():
    files = glob.glob(os.path.join(Config.Finetune.HDF5_DIR, '*.hdf5'))
    if len(files) == 0:
        raise FileNotFoundError(f"No HDF5 files found in {Config.Finetune.HDF5_DIR}")
        
    if Config.Finetune.RANDOMIZE_FILE_LIST:
        random.shuffle(files)
    else:
        files.sort()
        
    train_windows, train_labels, train_masks = [], [], []
    test_windows, test_labels, test_masks = [], [], []
    
    if Config.Finetune.RANDOM_MIX_MODE:
        target_files = files[:Config.Finetune.NUM_TRAIN_FILES + Config.Finetune.NUM_TEST_FILES]
        all_win, all_lbl, all_msk = [], [], []
        for f in target_files:
            w, l, m = extract_windows_from_file(f, is_train=True) 
            all_win.extend(w); all_lbl.extend(l); all_msk.extend(m)
            
        indices = np.arange(len(all_win))
        np.random.shuffle(indices)
        split = int(0.8 * len(indices))
        train_idx, test_idx = indices[:split], indices[split:]
        
        train_windows = [all_win[i] for i in train_idx]
        train_labels = [all_lbl[i] for i in train_idx]
        train_masks = [all_msk[i] for i in train_idx]
        test_windows = [all_win[i] for i in test_idx]
        test_labels = [all_lbl[i] for i in test_idx]
        test_masks = [all_msk[i] for i in test_idx]
    else:
        train_files = files[:Config.Finetune.NUM_TRAIN_FILES]
        test_files = files[Config.Finetune.NUM_TRAIN_FILES : Config.Finetune.NUM_TRAIN_FILES + Config.Finetune.NUM_TEST_FILES]
        
        for f in train_files:
            w, l, m = extract_windows_from_file(f, is_train=True)
            train_windows.extend(w); train_labels.extend(l); train_masks.extend(m)
        for f in test_files:
            w, l, m = extract_windows_from_file(f, is_train=False)
            test_windows.extend(w); test_labels.extend(l); test_masks.extend(m)
            
    unique_labels = sorted(list(set(train_labels + test_labels)))
    label2idx = {l: i for i, l in enumerate(unique_labels)}
    
    print(f">> Dataset fully loaded. Train samples: {len(train_windows)}, Test samples: {len(test_windows)}")
    return train_windows, train_labels, train_masks, test_windows, test_labels, test_masks, label2idx, unique_labels

# ==========================================
# 训练与评估逻辑
# ==========================================
def evaluate(model, loader, criterion, device):
    model.eval()
    correct, total = 0, 0
    total_loss = 0.0
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for x, y, v_mask in loader:
            x, y, v_mask = x.to(device), y.to(device), v_mask.to(device)
            outputs = model(x, valid_mask=v_mask)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            
    acc = correct / total
    avg_loss = total_loss / len(loader)
    return avg_loss, acc, all_targets, all_preds

def main():
    print(">> Preparing Data Pipeline...")
    train_win, train_lbl, train_msk, test_win, test_lbl, test_msk, label2idx, classes = prepare_data()
    
    train_dataset = EMGFinetuneDataset(train_win, train_lbl, train_msk, label2idx, is_train=True)
    test_dataset = EMGFinetuneDataset(test_win, test_lbl, test_msk, label2idx, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.Finetune.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.Finetune.BATCH_SIZE, shuffle=False)
    
    model = SleepFM_Finetune(num_classes=len(classes)).to(Config.DEVICE)
    
    # --- 调整：将 Dropout 从 0.5 降到适中的 0.4 ---
    if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
        model.classifier[0] = nn.Dropout(p=0.4)
        print(">> Adjusted Dropout rate to 0.4.")
    
    pretrained_path = "sleepfm_pretrained_ep50.pth" 
    if os.path.exists(pretrained_path):
        state_dict = torch.load(pretrained_path)
        encoder_dict = {k: v for k, v in state_dict.items() if k.startswith('encoder.')}
        model.load_state_dict(encoder_dict, strict=False)
        print(">> Successfully loaded Pre-trained Encoder weights!")
    else:
        print(">> No Pre-trained weights found. Training from scratch...")

    criterion = nn.CrossEntropyLoss()
    # --- 修复：降低 weight_decay 并提高学习率，帮助模型跳出全部预测 rest 的局部最优 ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    best_acc = 0.0
    best_weights = None
    patience = 5  
    patience_counter = 0
    
    print("\n>> Starting Fine-tuning...")
    for epoch in range(Config.Finetune.EPOCHS):
        model.train()
        total_loss = 0
        correct, total = 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.Finetune.EPOCHS}")
        
        for x, y, v_mask in pbar:
            x, y, v_mask = x.to(Config.DEVICE), y.to(Config.DEVICE), v_mask.to(Config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(x, valid_mask=v_mask)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        
        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, Config.DEVICE)
        print(f"Epoch {epoch+1} Completed | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"[*] New best accuracy {best_acc*100:.2f}%. Model weights saved.")
        else:
            patience_counter += 1
            print(f"[!] No improvement for {patience_counter} epochs.")
            
        if patience_counter >= patience:
            print(f"\n>> Early stopping triggered at Epoch {epoch+1}! Model has started overfitting.")
            break
            
    print("\n>> Training finished. Restoring best weights for final evaluation...")
    if best_weights is not None:
        model.load_state_dict(best_weights)
        
    final_loss, final_acc, targets, preds = evaluate(model, test_loader, criterion, Config.DEVICE)
    
    print(f"\n==================================================")
    print(f"🌟 Best Model Final Test Accuracy: {final_acc*100:.2f}%")
    print(f"==================================================\n")
    
    save_cm = True
    if save_cm:
        print("\nClassification Report:")
        print(classification_report(targets, preds, target_names=classes, zero_division=0))
        try:
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = ['Arial']
            cm = confusion_matrix(targets, preds)
            cm_pct = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
            plt.figure(figsize=(12, 10))
            ax = sns.heatmap(cm_pct, annot=True, fmt='.1%', cmap='Blues', xticklabels=classes, yticklabels=classes, annot_kws={"size": 30})
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=25); ax.set_yticklabels(ax.get_yticklabels(), fontsize=25)
            cbar = ax.collections[0].colorbar
            if cbar: cbar.ax.tick_params(labelsize=22)
            plt.title(f'Confusion Matrix (Final Acc: {final_acc*100:.2f}%)', fontsize=30, pad=25)
            plt.ylabel('True', fontsize=30, labelpad=15); plt.xlabel('Pred', fontsize=30, labelpad=15); plt.tight_layout(); plt.savefig('confusion_matrix.png', dpi=300)
            print("Confusion matrix saved to 'confusion_matrix.png'")
        except Exception as e: 
            print(f"Error plotting CM: {e}")

if __name__ == '__main__':
    main()
