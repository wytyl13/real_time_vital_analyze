#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/06/21 10:38
@Author  : weiyutao
@File    : lstm_engine.py
"""

"""
LSTM异常检测引擎 - 支持实时单条数据推理
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Union, Tuple, Optional, List
import joblib
import logging
from pathlib import Path


from .auto_encoders import RecurrentAutoencoder


class LSTMEngine:
    """
    LSTM异常检测引擎
    支持加载不同的LSTM模型和归一化器，进行实时单条数据异常检测
    """
    
    def __init__(
        self, 
        model_class,
        model_params: dict,
        seq_len: int = 20,
        n_features: int = 6,
        device: str = "auto",
        threshold: float = 0.5,
        threshold_method: str = "fixed",
        normalized_flag: int = 1
    ):
        """
        初始化LSTM引擎
        
        Args:
            model_class: 模型类（如RecurrentAutoencoder）
            model_params: 模型参数字典
            seq_len: 序列长度
            n_features: 特征数量
            device: 设备选择 ("auto", "cpu", "cuda")
            threshold: 异常检测阈值
            threshold_method: 阈值方法 ("fixed", "adaptive")
        """
        self.model_class = model_class
        self.model_params = model_params
        self.seq_len = seq_len
        self.n_features = n_features
        self.threshold = threshold
        self.threshold_method = threshold_method
        self.normalized_flag = normalized_flag
        
        # 设备配置
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # 初始化组件
        self.model = None
        self.scaler = None
        self.is_loaded = False
        
        # 日志配置
        self.logger = logging.getLogger(self.__class__.__name__)
        
        print(f"🤖 LSTM引擎初始化完成")
        print(f"   设备: {self.device}")
        print(f"   序列长度: {seq_len}, 特征数: {n_features}")
        print(f"   阈值: {threshold} ({threshold_method})")
    
    def load_model(self, model_path: str) -> bool:
        """
        加载训练好的模型
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            bool: 是否加载成功
        """
        try:
            # 检查文件是否存在
            if not Path(model_path).exists():
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
            # 加载检查点
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # 创建模型实例
            self.model = self.model_class(**self.model_params)
            
            # 加载模型状态
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            best_loss = checkpoint.get('best_loss', 'N/A')
            self.logger.info(f"模型加载成功! 最佳损失: {best_loss}")
            print(f"✅ 模型加载成功! 最佳损失: {best_loss}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {str(e)}")
            print(f"❌ 模型加载失败: {str(e)}")
            return False
    
    def load_scaler(self, scaler_path: Optional[str] = None, scaler_obj: Optional[StandardScaler] = None) -> bool:
        """
        加载归一化器
        
        Args:
            scaler_path: 归一化器文件路径
            scaler_obj: 直接传入的归一化器对象
            
        Returns:
            bool: 是否加载成功
        """
        try:
            if scaler_obj is not None:
                # 直接使用传入的scaler对象
                self.scaler = scaler_obj
                print("✅ 使用传入的归一化器对象")
                
            elif scaler_path is not None:
                # 从文件加载scaler
                if not Path(scaler_path).exists():
                    raise FileNotFoundError(f"归一化器文件不存在: {scaler_path}")
                
                self.scaler = joblib.load(scaler_path)
                print(f"✅ 从文件加载归一化器: {scaler_path}")
                
            else:
                # 创建默认的scaler（需要后续fit）
                self.scaler = StandardScaler()
                print("⚠️  创建了默认归一化器，需要后续使用数据进行fit")
            
            # 验证scaler
            if hasattr(self.scaler, 'n_features_in_') and self.scaler.n_features_in_ != self.n_features:
                print(f"⚠️  归一化器特征数({self.scaler.n_features_in_})与预期({self.n_features})不匹配")
            
            return True
            
        except Exception as e:
            self.logger.error(f"归一化器加载失败: {str(e)}")
            print(f"❌ 归一化器加载失败: {str(e)}")
            return False
    
    
    def setup(self, model_path: str, scaler_path: Optional[str] = None, scaler_obj: Optional[StandardScaler] = None) -> bool:
        """
        一键设置：加载模型和归一化器
        
        Args:
            model_path: 模型文件路径
            scaler_path: 归一化器文件路径
            scaler_obj: 直接传入的归一化器对象
            
        Returns:
            bool: 是否设置成功
        """
        print("🔧 开始设置LSTM引擎...")
        try:
            # 加载模型
            model_loaded = self.load_model(model_path)
            if not model_loaded:
                return False
            
            # 加载归一化器
            scaler_loaded = self.load_scaler(scaler_path, scaler_obj) if self.normalized_flag else None
            if not scaler_loaded and self.normalized_flag:
                return False
            
            self.is_loaded = True
            print("🎉 LSTM引擎设置完成，ready for inference!")
        except Exception as e:
            raise ValueError(f"Fial to load setup function! {str(e)}") from e
        return True
    
    def _validate_input(self, data: Union[list, tuple, np.ndarray]) -> np.ndarray:
        """
        验证和转换输入数据
        
        Args:
            data: 输入数据
            
        Returns:
            np.ndarray: 验证后的数据
        """
        # 转换为numpy数组
        if isinstance(data, (list, tuple)):
            data = np.array(data)
        elif not isinstance(data, np.ndarray):
            raise TypeError(f"不支持的数据类型: {type(data)}")
        
        # 检查维度
        if data.ndim == 1:
            if len(data) != self.n_features:
                raise ValueError(f"特征数不匹配: 期望{self.n_features}, 实际{len(data)}")
            data = data.reshape(1, -1)  # (1, n_features)
        elif data.ndim == 2:
            if data.shape[1] != self.n_features:
                raise ValueError(f"特征数不匹配: 期望{self.n_features}, 实际{data.shape[1]}")
        else:
            raise ValueError(f"不支持的数据维度: {data.ndim}, 期望1或2维")
        
        return data
    
    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """
        归一化数据
        
        Args:
            data: 原始数据 (n_samples, n_features)
            
        Returns:
            np.ndarray: 归一化后的数据
        """
        if self.scaler is None:
            raise ValueError("归一化器未加载")
        
        # 如果scaler未fit，抛出警告
        if not hasattr(self.scaler, 'mean_'):
            raise ValueError("归一化器未进行fit操作")
        
        return self.scaler.transform(data)
    
    def _compute_reconstruction_error(self, original: torch.Tensor, reconstructed: torch.Tensor) -> float:
        """
        计算重构误差
        
        Args:
            original: 原始数据
            reconstructed: 重构数据
            
        Returns:
            float: 重构误差
        """
        # 使用MSE作为重构误差
        # mse = torch.nn.functional.mse_loss(reconstructed, original, reduction='mean')
        # return mse.item()
        
        """
        组合误差计算，提高区分度
        """
        # MSE * 1000
        mse_scaled = torch.nn.functional.mse_loss(reconstructed, original, reduction='mean') * 1000
        
        # MAE * 100  
        mae_scaled = torch.nn.functional.l1_loss(reconstructed, original, reduction='mean') * 100
        
        # 返回组合分数
        return mse_scaled.item() + mae_scaled.item()



    def predict(
        self, 
        data: Union[list, tuple, np.ndarray], 
        return_details: bool = False,
    ) -> Union[bool, Tuple[bool, dict]]:
        """
        单条数据异常检测预测
        
        Args:
            data: 输入数据，形状为 (n_features,) 或 (seq_len, n_features)
            return_details: 是否返回详细信息
            
        Returns:
            bool: 是否异常 (return_details=False时)
            Tuple[bool, dict]: (是否异常, 详细信息) (return_details=True时)
        """
        if not self.is_loaded:
            raise RuntimeError("模型或归一化器未加载，请先调用setup()方法")
        
        try:
            # 1. 数据验证和转换
            data = self._validate_input(data)
            
            sequence_data = data
            
            # 3. 归一化
            
            normalized_data = self._normalize_data(sequence_data) if self.normalized_flag else sequence_data # (seq_len, n_features)

            
            # 4. 转换为PyTorch张量
            input_tensor = torch.FloatTensor(normalized_data).unsqueeze(0).to(self.device)  # (1, seq_len, n_features)
            
            # 5. 模型推理
            with torch.no_grad():
                reconstructed = self.model(input_tensor)  # (1, seq_len, n_features)
            
            # 6. 计算重构误差
            reconstruction_error = self._compute_reconstruction_error(input_tensor, reconstructed)
            
            # 7. 异常判断
            is_anomaly = reconstruction_error > self.threshold
            
            # 8. 准备详细信息
            # details = {
            #     'reconstruction_error': reconstruction_error,
            #     'threshold': self.threshold,
            #     'is_anomaly': is_anomaly,
            #     'input_shape': data.shape,
            #     'normalized_input': normalized_data,
            #     'reconstructed_output': reconstructed.cpu().numpy()
            # }
            
            # 9. 返回结果
            if return_details:
                return is_anomaly, reconstruction_error
            else:
                return is_anomaly
                
        except Exception as e:
            self.logger.error(f"预测过程出错: {str(e)}")
            raise RuntimeError(f"预测失败: {str(e)}")
    
    def update_threshold(self, new_threshold: float):
        """更新异常检测阈值"""
        self.threshold = new_threshold
        print(f"✅ 阈值已更新为: {new_threshold}")
    
    def get_status(self) -> dict:
        """获取引擎状态"""
        return {
            'model_loaded': self.model is not None,
            'scaler_loaded': self.scaler is not None,
            'is_ready': self.is_loaded,
            'device': str(self.device),
            'seq_len': self.seq_len,
            'n_features': self.n_features,
            'threshold': self.threshold,
            'threshold_method': self.threshold_method
        }


# 使用示例和测试代码
if __name__ == "__main__":
    
    def test_lstm_engine():
        """测试LSTM引擎"""
        print("🧪 测试LSTM引擎")
        print("=" * 50)
        
        # 1. 初始化引擎
        model_params = {
            'seq_len': 60,
            'n_features': 7,
            'embedding_dim': 128
        }
        
        engine = LSTMEngine(
            model_class=RecurrentAutoencoder,
            model_params=model_params,
            seq_len=60,
            n_features=7,
            threshold=0.5
        )
        engine.setup(model_path='/work/ai/WHOAMI/whoami/neural_network/rnn/checkpoint_epoch_148.pth', scaler_path='/work/ai/WHOAMI/whoami/neural_network/rnn/training_scaler.pkl')
        # 3. 设置引擎（这里跳过模型加载，因为没有真实的模型文件）
        
        # 4. 测试预测（需要真实模型才能工作）
        test_data = np.random.randn(60, 7)  # 单条数据
        print(f"测试数据形状: {test_data.shape}")
        print(f"测试数据: {test_data}")
        result = engine.predict(data=test_data, return_details=True)
        print(f"result: {result}")
    
    # 运行测试
    test_lstm_engine()