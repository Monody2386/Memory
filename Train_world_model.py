import importlib
import os
from typing import Iterable, Optional, Tuple, List
from shortmemory import ScoredTensorQueue, short_memory
import torch


def _wm():
    """动态导入 World_model 模块"""
    return importlib.import_module("World_model")


def train_world_model(
    samples: Iterable[Tuple[torch.Tensor, int, torch.Tensor]],
    save_dir: str = "world_models",
    load_dir: Optional[str] = None,
    num_epochs: int = 1,
    device: str = "cpu"
):
    """
    训练世界模型
    
    Args:
        samples: 训练样本迭代器，每个样本为 (input_2d, action_type, target_action)
                 - input_2d: (noun_dim+action_dim, seq_len) 输入序列
                 - action_type: int 模型索引 (0, 1, 2)
                 - target_action: (action_dim,) 目标的下一个action
        save_dir: 保存模型的目录
        load_dir: 加载预训练模型的目录
        num_epochs: 训练轮数
        device: 计算设备 ('cpu' 或 'cuda')
    """
    wm_module = _wm()
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建世界模型实例
    world_model = wm_module.WorldModel(
        noun_dim=wm_module.noun_dim,
        action_dim=wm_module.action_dim,
        attention_dim=wm_module.attention_dim,
        value_dim=wm_module.value_dim,
        hidden_dim=wm_module.hidden_dim
    )
    world_model = world_model.to(device)
    
    # 加载预训练模型（如果存在）
    if load_dir is not None:
        model_path = os.path.join(load_dir, "world_model.pt")
        if os.path.exists(model_path):
            world_model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"加载预训练模型: {model_path}")
    
    # 创建优化器
    optimizer = torch.optim.Adam(world_model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    
    # 训练循环
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        # 转换为列表以支持多轮epoch
        samples_list = list(samples)
        
        for input_2d, action_type, target_action in samples_list:
            # 移动数据到设备
            input_2d = input_2d.to(device)
            target_action = target_action.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            pred_action = world_model(input_2d, action_type)  # (action_dim,)
            
            # 计算损失
            loss = loss_fn(pred_action, target_action)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.6f}")
    
    # 保存模型
    model_path = os.path.join(save_dir, "world_model.pt")
    torch.save(world_model.state_dict(), model_path)
    print(f"模型已保存: {model_path}")
    
    return world_model


def train_world_model_with_validation(
    train_samples: Iterable[Tuple[torch.Tensor, int, torch.Tensor]],
    val_samples: Optional[Iterable[Tuple[torch.Tensor, int, torch.Tensor]]] = None,
    save_dir: str = "world_models",
    load_dir: Optional[str] = None,
    num_epochs: int = 10,
    device: str = "cpu",
    patience: int = 5
) -> dict:
    """
    带验证集的世界模型训练
    
    Args:
        train_samples: 训练样本迭代器
        val_samples: 验证样本迭代器（可选）
        save_dir: 保存模型的目录
        load_dir: 加载预训练模型的目录
        num_epochs: 最大训练轮数
        device: 计算设备
        patience: 早停耐心值
    
    Returns:
        dict 包含：
            - 'model': 训练好的模型
            - 'train_losses': 训练损失列表
            - 'val_losses': 验证损失列表
            - 'best_epoch': 最佳epoch
    """
    wm_module = _wm()
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建世界模型实例
    world_model = wm_module.WorldModel(
        noun_dim=wm_module.noun_dim,
        action_dim=wm_module.action_dim,
        attention_dim=wm_module.attention_dim,
        value_dim=wm_module.value_dim,
        hidden_dim=wm_module.hidden_dim
    )
    world_model = world_model.to(device)
    
    # 加载预训练模型（如果存在）
    if load_dir is not None:
        model_path = os.path.join(load_dir, "world_model.pt")
        if os.path.exists(model_path):
            world_model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"加载预训练模型: {model_path}")
    
    # 创建优化器
    optimizer = torch.optim.Adam(world_model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    # 转换为列表以支持多轮epoch
    train_samples_list = list(train_samples)
    val_samples_list = list(val_samples) if val_samples is not None else []
    
    # 训练循环
    for epoch in range(num_epochs):
        # ===== 训练阶段 =====
        world_model.train()
        epoch_loss = 0.0
        batch_count = 0
        
        for input_2d, action_type, target_action in train_samples_list:
            input_2d = input_2d.to(device)
            target_action = target_action.to(device)
            
            optimizer.zero_grad()
            pred_action = world_model(input_2d, action_type)
            loss = loss_fn(pred_action, target_action)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        avg_train_loss = epoch_loss / batch_count if batch_count > 0 else 0
        train_losses.append(avg_train_loss)
        
        # ===== 验证阶段 =====
        if val_samples_list:
            world_model.eval()
            val_epoch_loss = 0.0
            val_batch_count = 0
            
            with torch.no_grad():
                for input_2d, action_type, target_action in val_samples_list:
                    input_2d = input_2d.to(device)
                    target_action = target_action.to(device)
                    
                    pred_action = world_model(input_2d, action_type)
                    loss = loss_fn(pred_action, target_action)
                    
                    val_epoch_loss += loss.item()
                    val_batch_count += 1
            
            avg_val_loss = val_epoch_loss / val_batch_count if val_batch_count > 0 else 0
            val_losses.append(avg_val_loss)
            
            print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
            
            # 早停逻辑
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_epoch = epoch + 1
                
                # 保存最佳模型
                best_model_path = os.path.join(save_dir, "world_model_best.pt")
                torch.save(world_model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早停触发在 epoch {epoch+1}，最佳epoch: {best_epoch}")
                    break
        else:
            print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.6f}")
    
    # 保存最终模型
    final_model_path = os.path.join(save_dir, "world_model.pt")
    torch.save(world_model.state_dict(), final_model_path)
    print(f"模型已保存: {final_model_path}")
    
    return {
        'model': world_model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_epoch': best_epoch
    }


if __name__ == "__main__":
    short_memory = ScoredTensorQueue()
    