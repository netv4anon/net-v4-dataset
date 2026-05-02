"""
NYC-EMS-Traffic V4 (NET-V4) Data Loader
========================================
Loads NET-V4 dataset from NPZ files.

Source: NYC_EMS_Traffic_V4.npz + NYC_EMS_Traffic_V4_graph.npz
Specs:
- 33,076 samples (20,000 EV + 13,076 control)
- 363-node graph (mean degree 39.6)
- 24 time steps at 15-min resolution
- Supports causal inference (treated vs. control)
- Standard (B, T, N, C) tensor format
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional, Tuple


class NETV4Dataset(Dataset):
    """
    NYC-EMS-Traffic V4 PyTorch Dataset
    
    每个样本包含:
    - x: (T_in, N_spatial, C) 历史输入 (中心节点 + 8 邻居的多通道特征)
    - y: (T_out, N_spatial) 未来流量目标
    - ev_signal: (T_in, N_spatial, 4) EV 控制信号
    - is_ev: bool 是否为 EV 事件
    - metadata: dict 额外信息
    """
    def __init__(self, data: Dict[str, np.ndarray], split: str = 'train',
                 input_steps: int = 12, pred_steps: int = 12,
                 normalize: bool = True, 
                 train_stats: Optional[Dict] = None):
        super().__init__()
        self.input_steps = input_steps
        self.pred_steps = pred_steps
        self.split = split
        
        # 数据划分索引
        n_total = data['volume_sequences'].shape[0]
        n_train = int(n_total * 0.666)
        n_val = int(n_total * 0.167)
        
        if split == 'train':
            idx_start, idx_end = 0, n_train
        elif split == 'val':
            idx_start, idx_end = n_train, n_train + n_val
        else:  # test
            idx_start, idx_end = n_train + n_val, n_total
            
        sl = slice(idx_start, idx_end)
        
        # 主序列: (N_samples, 24)
        self.volume = data['volume_sequences'][sl].astype(np.float32)
        # 邻居序列: (N_samples, 8, 24)
        self.spatial_volumes = data['spatial_volumes'][sl].astype(np.float32)
        # EV 掩码: (N_samples, 24)
        self.ev_masks = data['ev_masks'][sl].astype(np.float32)
        # 事件标签
        self.is_ev = data['is_ev_event'][sl].astype(np.float32)
        # 严重度
        self.severity = data['severity_scores'][sl].astype(np.float32)
        # 空间距离
        self.spatial_dist = data['spatial_distances'][sl].astype(np.float32)
        # 时间编码
        self.hours = data['incident_hours'][sl].astype(np.float32)
        self.dows = data['incident_dows'][sl].astype(np.float32)
        # 响应时间
        self.dispatch_sec = data['dispatch_seconds'][sl].astype(np.float32)
        self.response_sec = data['response_seconds'][sl].astype(np.float32)
        # 基线流量
        self.baseline = data['baseline_volumes'][sl].astype(np.float32)
        
        # 归一化 (仅基于训练集统计)
        self.normalize = normalize
        if normalize:
            if train_stats is None:
                # 计算训练集统计量
                train_vol = data['volume_sequences'][:n_train]
                train_spatial = data['spatial_volumes'][:n_train]
                self.vol_mean = float(np.mean(train_vol))
                self.vol_std = float(np.std(train_vol)) + 1e-6
                self.spatial_mean = float(np.mean(train_spatial))
                self.spatial_std = float(np.std(train_spatial)) + 1e-6
            else:
                self.vol_mean = train_stats['vol_mean']
                self.vol_std = train_stats['vol_std']
                self.spatial_mean = train_stats['spatial_mean']
                self.spatial_std = train_stats['spatial_std']
        
        self.n_samples = self.volume.shape[0]
        
    def get_stats(self) -> Dict:
        return {
            'vol_mean': self.vol_mean,
            'vol_std': self.vol_std,
            'spatial_mean': self.spatial_mean,
            'spatial_std': self.spatial_std,
        }
        
    def __len__(self):
        return self.n_samples
    
    def _normalize_vol(self, v):
        if self.normalize:
            return (v - self.vol_mean) / self.vol_std
        return v
    
    def _normalize_spatial(self, v):
        if self.normalize:
            return (v - self.spatial_mean) / self.spatial_std
        return v
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        # 中心节点流量 (24,)
        vol = self.volume[idx]  # (24,)
        # 邻居流量 (K, 24) - K 从数据动态读取
        spatial = self.spatial_volumes[idx]  # (K, 24)
        
        # 构建空间维度: N_spatial = 1(中心) + K(邻居)
        all_volumes = np.concatenate([vol[np.newaxis, :], spatial], axis=0)  # (1+K, 24)
        
        # 时间分割: 前 input_steps 为输入, 后 pred_steps 为目标
        T = all_volumes.shape[1]  # 24
        t_split = min(self.input_steps, T - self.pred_steps)
        
        x_vol = self._normalize_vol(all_volumes[:, :t_split])       # (9, T_in)
        y_vol = all_volumes[:, t_split:t_split+self.pred_steps]      # (9, T_out)
        
        # EV 信号构建: (T_in, 9, 4)
        ev_mask_t = self.ev_masks[idx, :t_split]  # (T_in,)
        severity = np.full((t_split,), self.severity[idx])
        
        # 空间距离: 中心=0, 邻居=spatial_dist
        sp_dist = np.concatenate([[0.0], self.spatial_dist[idx]])  # (9,)
        
        # 响应时间归一化代理
        resp_norm = np.full((t_split,), min(self.response_sec[idx] / 600.0, 1.0))
        
        # 时间编码
        hour = self.hours[idx]
        dow = self.dows[idx]
        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)
        dow_sin = np.sin(2 * np.pi * dow / 7.0)
        dow_cos = np.cos(2 * np.pi * dow / 7.0)
        
        # 构建完整的 12 通道输入: (T_in, N_sp, 12)
        T_in = t_split
        N_sp = all_volumes.shape[0]  # 1 + K_NEIGHBORS (动态)
        
        features = np.zeros((T_in, N_sp, 12), dtype=np.float32)
        
        # Ch0: 归一化流量
        features[:, :, 0] = x_vol.T  # (T_in, 9)
        
        # Ch1: EV 掩码 (所有空间节点共享)
        features[:, :, 1] = ev_mask_t[:, np.newaxis]
        
        # Ch2-5: 时间编码
        features[:, :, 2] = hour_sin
        features[:, :, 3] = hour_cos
        features[:, :, 4] = dow_sin
        features[:, :, 5] = dow_cos
        
        # Ch6: profile deviation (根据基线)
        baseline = self.baseline[idx]
        if baseline > 0:
            features[:, :, 6] = (x_vol.T * self.vol_std + self.vol_mean - baseline) / (baseline + 1e-6) if self.normalize else (x_vol.T - baseline) / (baseline + 1e-6)
        
        # Ch7: severity
        features[:, :, 7] = self.severity[idx]
        
        # Ch8: spatial distance
        features[:, :, 8] = sp_dist[np.newaxis, :]  # broadcast over time
        
        # Ch9: node index (归一化)
        features[:, :, 9] = np.arange(N_sp)[np.newaxis, :] / N_sp
        
        # Ch10: speed proxy (=1 - congestion)
        features[:, :, 10] = 1.0 - np.clip(features[:, :, 0], 0, 1)
        
        # Ch11: congestion index
        features[:, :, 11] = np.clip(features[:, :, 0], 0, 1)
        
        # EV 控制信号: (T_in, 9, 4) = [ev_mask, severity, spatial_dist, response_norm]
        ev_signal = np.zeros((T_in, N_sp, 4), dtype=np.float32)
        ev_signal[:, :, 0] = ev_mask_t[:, np.newaxis]
        ev_signal[:, :, 1] = self.severity[idx]
        ev_signal[:, :, 2] = sp_dist[np.newaxis, :]
        ev_signal[:, :, 3] = resp_norm[:, np.newaxis]
        
        return {
            'x': torch.from_numpy(features),              # (T_in, 9, 12)
            'y': torch.from_numpy(y_vol.T),                # (T_out, 9)
            'ev_signal': torch.from_numpy(ev_signal),      # (T_in, 9, 4)
            'is_ev': torch.tensor(self.is_ev[idx]),        # scalar
            'severity': torch.tensor(self.severity[idx]),
            'y_raw_center': torch.from_numpy(y_vol[0]),    # (T_out,) 中心节点原始目标
        }


class NETV4GraphDataset(Dataset):
    """
    全图级别的 V4 数据集
    将 363 节点图上的时序数据组织为标准 (B, T, N, C) 格式
    
    适用于：与 METR-LA/PEMS-BAY 对比的标准化实验
    """
    def __init__(self, data: Dict[str, np.ndarray], graph: Dict[str, np.ndarray],
                 split: str = 'train', input_steps: int = 12, pred_steps: int = 12,
                 train_stats: Optional[Dict] = None):
        super().__init__()
        self.input_steps = input_steps
        self.pred_steps = pred_steps
        
        n_total = data['volume_sequences'].shape[0]
        n_train = int(n_total * 0.666)
        n_val = int(n_total * 0.167)
        
        if split == 'train':
            sl = slice(0, n_train)
        elif split == 'val':
            sl = slice(n_train, n_train + n_val)
        else:
            sl = slice(n_train + n_val, n_total)
        
        # 提取每个样本的 segment_id 并映射到全图节点
        self.segment_ids = data['segment_ids'][sl]
        self.neighbor_ids = data['neighbor_seg_ids'][sl]
        self.volumes = data['volume_sequences'][sl].astype(np.float32)
        self.spatial_vols = data['spatial_volumes'][sl].astype(np.float32)
        self.ev_masks = data['ev_masks'][sl].astype(np.float32)
        self.is_ev = data['is_ev_event'][sl].astype(np.float32)
        self.severity = data['severity_scores'][sl].astype(np.float32)
        self.spatial_dist = data['spatial_distances'][sl].astype(np.float32)
        self.hours = data['incident_hours'][sl].astype(np.float32)
        self.dows = data['incident_dows'][sl].astype(np.float32)
        self.response_sec = data['response_seconds'][sl].astype(np.float32)
        self.baseline = data['baseline_volumes'][sl].astype(np.float32)
        
        self.adj = graph['adjacency'].astype(np.float32) if 'adjacency' in graph else None
        self.num_nodes = int(graph.get('adjacency', np.zeros((363, 363))).shape[0])
        
        # 归一化
        if train_stats is None:
            train_vol = data['volume_sequences'][:n_train]
            self.vol_mean = float(np.mean(train_vol))
            self.vol_std = float(np.std(train_vol)) + 1e-6
        else:
            self.vol_mean = train_stats['vol_mean']
            self.vol_std = train_stats['vol_std']
            
        self.n_samples = self.volumes.shape[0]
        
    def get_stats(self):
        return {'vol_mean': self.vol_mean, 'vol_std': self.vol_std}
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        T = 24
        t_split = min(self.input_steps, T - self.pred_steps)
        
        vol = (self.volumes[idx] - self.vol_mean) / self.vol_std  # (24,)
        
        x_vol = vol[:t_split]    # (T_in,)
        y_vol = self.volumes[idx, t_split:t_split+self.pred_steps]  # (T_out,) 未归一化
        
        ev_mask = self.ev_masks[idx, :t_split]
        
        # 构建简化特征: (T_in, 4) = [volume, ev_mask, hour_enc, severity]
        hour = self.hours[idx]
        features = np.stack([
            x_vol,
            ev_mask,
            np.full(t_split, np.sin(2*np.pi*hour/24)),
            np.full(t_split, self.severity[idx]),
        ], axis=-1).astype(np.float32)  # (T_in, 4)
        
        return {
            'x': torch.from_numpy(features),          # (T_in, 4)
            'y': torch.from_numpy(y_vol),              # (T_out,)
            'is_ev': torch.tensor(self.is_ev[idx]),
            'segment_id': torch.tensor(int(self.segment_ids[idx])),
            'severity': torch.tensor(self.severity[idx]),
        }


def load_net_v4(data_dir: str, mode: str = 'local') -> Tuple[Dict, Dict]:
    """
    加载 NET V4 数据集
    
    mode:
        'local': 局部模式 (每样本 9 节点子图)
        'graph': 全图模式 (363 节点)
    
    returns: (data_dict, graph_dict)
    """
    # 主数据文件
    data_path = os.path.join(data_dir, 'NYC_EMS_Traffic_V4.npz')
    graph_path = os.path.join(data_dir, 'NYC_EMS_Traffic_V4_graph.npz')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"V4 数据文件未找到: {data_path}")
    
    data = dict(np.load(data_path, allow_pickle=True))
    
    graph = {}
    if os.path.exists(graph_path):
        graph = dict(np.load(graph_path, allow_pickle=True))
    
    print(f"[NET-V4] 加载完成:")
    print(f"  样本数: {data['volume_sequences'].shape[0]}")
    print(f"  时间步: {data['volume_sequences'].shape[1]}")
    print(f"  EV事件: {int(data['is_ev_event'].sum())}")
    print(f"  对照组: {int((1 - data['is_ev_event']).sum())}")
    if 'adjacency' in graph:
        adj = graph['adjacency']
        print(f"  图节点: {adj.shape[0]}, 边数: {np.count_nonzero(adj)}")
    
    return data, graph


def create_net_v4_dataloaders(data_dir: str, batch_size: int = 64,
                               input_steps: int = 12, pred_steps: int = 12,
                               num_workers: int = 0) -> Dict:
    """
    创建 NET V4 的 PyTorch DataLoader
    
    returns: {
        'train': DataLoader,
        'val': DataLoader, 
        'test': DataLoader,
        'adj': np.ndarray,
        'stats': dict,
        'num_nodes': int,
    }
    """
    data, graph = load_net_v4(data_dir)
    
    # 创建训练集 (获取归一化统计量)
    train_ds = NETV4Dataset(data, 'train', input_steps, pred_steps)
    stats = train_ds.get_stats()
    
    # 创建验证集和测试集 (使用训练集统计量)
    val_ds = NETV4Dataset(data, 'val', input_steps, pred_steps, train_stats=stats)
    test_ds = NETV4Dataset(data, 'test', input_steps, pred_steps, train_stats=stats)
    
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    loaders = {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                           num_workers=num_workers, pin_memory=True, drop_last=True),
        'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True),
        'test': DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True),
    }
    
    adj = graph.get('adjacency', np.eye(9))
    
    return {
        **loaders,
        'adj': adj,
        'stats': stats,
        'num_nodes': 9,  # 局部子图模式
        'num_global_nodes': adj.shape[0] if len(adj.shape) == 2 else 9,
    }
