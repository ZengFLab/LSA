import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils.custom_mlp import MLP
from .utils.utils import CustomDataset, tensor_to_numpy, convert_to_tensor


import os
import random
import numpy as np
from tqdm import tqdm

from typing import Literal,Callable

import warnings
warnings.filterwarnings("ignore")

import dill as pickle
import gzip
from packaging.version import Version
torch_version = torch.__version__


def set_random_seed(seed):
    # Set seed for PyTorch
    torch.manual_seed(seed)
    
    # If using CUDA, set the seed for CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups.
    
    # Set seed for NumPy
    np.random.seed(seed)
    
    # Set seed for Python's random module
    random.seed(seed)



class LatentAligner(nn.Module):
    """
    """
    def __init__(self,
                 decoder_a: Callable, 
                 decoder_b: Callable, 
                 z_a_dim: int, 
                 z_b_dim: int,
                 hidden_layers: list = [512,512],
                 hidden_layer_activation: Literal['relu','softplus','leakyrelu','linear'] = 'relu',
                 use_cuda: bool = True,
                 seed: int = 42,
                 dtype = torch.float32, # type: ignore
                 ):
        super().__init__()

        self.hidden_layers = hidden_layers
        self.decoder_hidden_layers = hidden_layers[::-1]
        self.use_cuda = use_cuda
        self.options = None
        self.dtype = dtype
        self.hidden_layer_activation = hidden_layer_activation
        
        self.decoder_a = decoder_a 
        self.decoder_b = decoder_b
        self.z_a_dim = z_a_dim
        self.z_b_dim = z_b_dim
        self.seed = seed
        
        set_random_seed(seed)
        self.setup_networks()

    def setup_networks(self):
        hidden_sizes = self.hidden_layers

        if self.hidden_layer_activation == 'relu':
            activate_fct = nn.ReLU
        elif self.hidden_layer_activation == 'softplus':
            activate_fct = nn.Softplus
        elif self.hidden_layer_activation == 'leakyrelu':
            activate_fct = nn.LeakyReLU
        elif self.hidden_layer_activation == 'linear':
            activate_fct = nn.Identity

        self.map_a_to_b = MLP(
            [self.z_a_dim] + hidden_sizes + [self.z_b_dim],
            activation=activate_fct,
            output_activation=None,
            use_cuda=self.use_cuda,
        )
            
        if self.use_cuda:
            self.cuda()

    def get_device(self):
        return next(self.parameters()).device

    def forward(self, z_a, batch_size: int = 1024, show_progress: bool = True):
        """
        """
        ##################
        z_a = convert_to_tensor(z_a, device='cpu')
        dataset = CustomDataset(z_a)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        Z = []
        with tqdm(total=len(dataloader), disable=not show_progress, desc='', unit='batch') as pbar:
            for Z_batch, _ in dataloader:
                Z_batch = Z_batch.to(self.get_device())
                z_b = self.map_a_to_b(Z_batch)
                Z.append(tensor_to_numpy(z_b))
                pbar.update(1)

        Z = np.concatenate(Z)
        
        return Z

    def fit(self, 
            z_a,
        num_epochs: int = 100, 
        learning_rate: float = 0.0001, 
        batch_size: int = 64, 
        algo: Literal['adam','rmsprop','adamw'] = 'adam', 
        reduction: str = 'mean',
        beta_1: float = 0.9, 
        weight_decay: float = 0.005, 
        decay_rate: float = 0.9,
        use_jax: bool = False,
        show_progress: bool = True,
        # Early stopping 相关参数
        patience: int = 10,
        min_delta: float = 1e-6,
        restore_best_weights: bool = True):
        """
        训练映射网络
        
        参数:
            num_samples: 训练样本数
            num_epochs: 最大训练轮数
            learning_rate: 学习率
            batch_size: 批次大小
            algo: 优化器类型
            beta_1: Adam的beta1参数
            weight_decay: 权重衰减
            decay_rate: RMSprop的衰减率（未使用，保留接口）
            use_jax: 是否使用JAX（未使用，保留接口）
            show_progress: 是否显示进度条
            patience: Early stopping的耐心值
            min_delta: 被视为改进的最小变化阈值
            restore_best_weights: 是否恢复最佳权重
        """            
        z_a = convert_to_tensor(z_a, dtype=self.dtype, device='cpu')

        dataset = CustomDataset(z_a)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # ---------- 优化器检查与设置 ----------
        # 检查1: algo参数有效性
        valid_algos = ['adam', 'rmsprop', 'adamw']
        if algo.lower() not in valid_algos:
            raise ValueError(f"Optimizer must be one of {valid_algos}, received: {algo}")
        
        # 检查2: 学习率合理性
        if learning_rate <= 0:
            raise ValueError(f"Learning rate must be positive: {learning_rate}")
        
        # 检查3: beta参数有效性
        if not (0 <= beta_1 < 1):
            raise ValueError(f"beta_1 should be in [0,1]: {beta_1}")
        
        # 根据不同优化器设置不同的参数
        if algo.lower() == 'rmsprop':
            optim_params = {
                'lr': learning_rate,
                'alpha': decay_rate,  # RMSprop用alpha代替beta
                'weight_decay': weight_decay
            }
        elif algo.lower() == 'adam':
            optim_params = {
                'lr': learning_rate,
                'betas': (beta_1, 0.999),
                'weight_decay': weight_decay
            }
        elif algo.lower() == 'adamw':
            optim_params = {
                'lr': learning_rate,
                'betas': (beta_1, 0.999),
                'weight_decay': weight_decay
            }
        
        # 初始化优化器
        if algo.lower() == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.map_a_to_b.parameters(), **optim_params)
        elif algo.lower() == 'adam':
            optimizer = torch.optim.Adam(self.map_a_to_b.parameters(), **optim_params)
        elif algo.lower() == 'adamw':
            optimizer = torch.optim.AdamW(self.map_a_to_b.parameters(), **optim_params)
        
        # 检查4: 验证优化器确实包含了所有参数
        param_count = sum(p.numel() for p in self.map_a_to_b.parameters())
        optim_param_count = sum(p.numel() for group in optimizer.param_groups for p in group['params'])
        if param_count != optim_param_count:
            print(f"Warn: model parameters ({param_count}) and optimizer parameters ({optim_param_count}) do not match")
        
        # ---------- 损失函数设置 ----------
        lossfunc = nn.MSELoss(reduction=reduction)
        
        # ---------- Early Stopping 初始化 ----------
        best_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        best_state = None
        early_stopped = False
        
        # ---------- 训练历史记录 ----------
        history = {
            'train_loss': [],
            'best_loss': None,
            'best_epoch': None,
            'stopped_epoch': None
        }
        
        # ---------- 训练循环 ----------
        with tqdm(total=num_epochs, disable=not show_progress, 
                desc='Training', unit='epoch') as pbar:
            for epoch in range(num_epochs):
                # 设置为训练模式
                self.map_a_to_b.train()
                
                epoch_loss_sum = 0.0
                num_batches = 0
                
                for batch_z_a, _ in dataloader:
                    batch_z_a = batch_z_a.to(self.get_device())
                    
                    # 前向传播
                    batch_z_b_pred = self.map_a_to_b(batch_z_a)
                    
                    # 解码到特征空间
                    batch_x_a = self.decoder_a(batch_z_a)
                    batch_x_b = self.decoder_b(batch_z_b_pred)
                    
                    # 计算损失
                    batch_loss = lossfunc(batch_x_a, batch_x_b)
                    
                    # 反向传播
                    optimizer.zero_grad()
                    batch_loss.backward()
                    
                    # 检查5: 梯度裁剪（防止梯度爆炸）
                    #torch.nn.utils.clip_grad_norm_(self.map_a_to_b.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    epoch_loss_sum += batch_loss.item()
                    num_batches += 1
                
                # 计算本epoch的平均损失
                avg_epoch_loss = epoch_loss_sum / num_batches
                history['train_loss'].append(avg_epoch_loss)
                
                # ---------- 更新进度条显示 ----------
                pbar.set_postfix({
                    'loss': f'{avg_epoch_loss:.6f}',
                    'best': f'{best_loss:.6f}' if best_loss != float('inf') else 'N/A',
                    'patience': f'{patience_counter}/{patience}'
                })
                pbar.update(1)
                
                # ---------- Early Stopping 检查 ----------
                current_loss = avg_epoch_loss
                
                if current_loss < best_loss - min_delta:
                    # 有显著改进
                    improvement = best_loss - current_loss
                    best_loss = current_loss
                    best_epoch = epoch
                    patience_counter = 0
                    
                    # 保存最佳权重
                    if restore_best_weights:
                        best_state = {
                            'epoch': epoch,
                            'model_state_dict': self.map_a_to_b.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': best_loss,
                        }
                    
                    if show_progress:
                        pbar.set_postfix_str(f'loss={avg_epoch_loss:.6f}, '
                                            f'improved by {improvement:.6f}')
                else:
                    # 没有显著改进
                    patience_counter += 1
                    
                    if patience_counter >= patience:
                        if show_progress:
                            print(f"\nEarly stopping at epoch {epoch+1}")
                            print(f"Best loss: {best_loss:.6f} at epoch {best_epoch+1}")
                        
                        early_stopped = True
                        history['stopped_epoch'] = epoch
                        break
            
            # ---------- 训练结束处理 ----------
            if early_stopped:
                print(f"Training stopped early due to no improvement for {patience} epochs.")
            else:
                print(f"Training completed all {num_epochs} epochs.")
            
            # 恢复最佳权重
            if restore_best_weights and best_state is not None:
                self.map_a_to_b.load_state_dict(best_state['model_state_dict'])
                print(f"Restored best weights from epoch {best_epoch+1} with loss {best_loss:.6f}")
            
            # 更新历史记录
            history['best_loss'] = best_loss
            history['best_epoch'] = best_epoch
            
            # 打印训练总结
            print(f"\n{'='*50}")
            print(f"Training Summary:")
            print(f"{'='*50}")
            print(f"Total epochs trained: {epoch+1}")
            print(f"Best loss: {best_loss:.6f} (epoch {best_epoch+1})")
            print(f"Final loss: {avg_epoch_loss:.6f}")
            print(f"Optimizer: {algo.upper()}")
            print(f"Learning rate: {learning_rate}")
            print(f"Weight decay: {weight_decay}")
            if early_stopped:
                print(f"Early stopped: Yes (patience={patience})")
            else:
                print(f"Early stopped: No")
            print(f"{'='*50}")
                                
    @classmethod
    def save_model(cls, model, file_path, compression=False):
        """Save the model to the specified file path."""
        file_path = os.path.abspath(file_path)

        model.eval()
        if compression:
            with gzip.open(file_path, 'wb') as pickle_file:
                pickle.dump(model, pickle_file)
        else:
            with open(file_path, 'wb') as pickle_file:
                pickle.dump(model, pickle_file)

        print(f'Model saved to {file_path}')

    @classmethod
    def load_model(cls, file_path):
        """Load the model from the specified file path and return an instance."""
        print(f'Model loaded from {file_path}')

        file_path = os.path.abspath(file_path)
        if file_path.endswith('gz'):
            with gzip.open(file_path, 'rb') as pickle_file:
                model = pickle.load(pickle_file)
        else:
            with open(file_path, 'rb') as pickle_file:
                model = pickle.load(pickle_file)
        
        return model



