import torch
import torch.nn as nn
import torch.nn.functional as F

attention_dim = 100
action_dim = 80
noun_dim = 40
value_dim = 100
hidden_dim = 50
class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Event_Attention(nn.Module):
    def __init__(self,noun_dim, action_dim, attention_dim, value_dim):
        super(Event_Attention,self).__init__()
        self.Wq=nn.Linear(noun_dim + action_dim, attention_dim)
        self.Wk=nn.Linear(noun_dim + action_dim, attention_dim)
        self.Wv=nn.Linear(noun_dim + action_dim, value_dim)
        self.scale = attention_dim ** 0.5
        
    def forward(self,input):
        
        input_prefix = input[:,:-1,:]  
        input_suffix = input[:,-1,:]    

        q = self.Wq(input_suffix)
        k = self.Wk(input_prefix)
        v = self.Wv(input_prefix)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attention_weight = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weight, v)
        output = output + self.W_v(input_suffix)
        return output
    
class ActionModel(nn.Module):
    def __init__(self, noun_dim, action_dim, attention_dim, value_dim, hidden_dim):
        super(ActionModel, self).__init__()

        # ===== Attention 部分 =====
        self.Wq = nn.Linear(noun_dim + action_dim, attention_dim)
        self.Wk = nn.Linear(noun_dim + action_dim, attention_dim)
        self.Wv = nn.Linear(noun_dim + action_dim, value_dim)
        self.scale = attention_dim ** 0.5

        # ===== FFN  =====
        self.action_predict = nn.Sequential(
                nn.Linear(value_dim, hidden_dim, bias=False),
                nn.Linear(hidden_dim, action_dim, bias=False)
            )

    def forward(self, x):
        input_prefix = x[:, :-1, :]
        input_suffix = x[:, -1, :]

        # ===== Attention =====
        q = self.Wq(input_suffix)      
        k = self.Wk(input_prefix)        
        v = self.Wv(input_prefix)        

        scores = torch.matmul(q.unsqueeze(1), k.transpose(-2, -1)) / self.scale
        attention_weight = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weight, v).squeeze(1) 
        output = output + self.Wv(input_suffix)
            # ===== FFN =====
        output = self.action_predict(output)
        return output 
    

Action_models = nn.ModuleList([
    ActionModel(noun_dim, action_dim, attention_dim, value_dim, hidden_dim),
    ActionModel(noun_dim, action_dim, attention_dim, value_dim, hidden_dim),
    ActionModel(noun_dim, action_dim, attention_dim, value_dim, hidden_dim)
])

Action_list = ['action1', 'action2', 'action3']

optimizer = torch.optim.Adam([
    {'params': Action_models[0].parameters(), 'lr': 1e-3},
    {'params': Action_models[1].parameters(), 'lr': 1e-4},
    # ...
    
])

def train_action_model(input_data, target_action, action_index):
    for step in range(100):
        optimizer.zero_grad()
        action_pred = Action_models[action_index](input_data)
        loss = nn.MSELoss()(action_pred, target_action)
        loss.backward()
        optimizer.step()


class WorldModel(nn.Module):
    """
    世界模型 - 综合管理多个Action模型
    
    Forward 输入：(input_2d, action_type)
    Forward 输出：new_action_type
    
    核心逻辑：
    1. 接收二维输入 input_2d (noun_dim+action_dim, seq_len) 和 action_type (模型索引)
    2. 将输入转换为 (1, seq_len, noun_dim+action_dim) 格式
    3. 通过指定的ActionModel前向传播得到预测的action
    4. 返回预测的新action (action_dim,)
    """
    
    def __init__(self, noun_dim, action_dim, attention_dim, value_dim, hidden_dim):
        super(WorldModel, self).__init__()
        self.noun_dim = noun_dim
        self.action_dim = action_dim
        self.attention_dim = attention_dim
        self.value_dim = value_dim
        self.hidden_dim = hidden_dim
        
        # 使用预定义的Action models
        self.action_models = Action_models
        self.action_list = Action_list
        self.model_count = len(self.action_models)
    
    def forward(self, input_2d: torch.Tensor, action_type: int) -> torch.Tensor:
        """
        世界模型前向传播
        
        Args:
            input_2d: 形状 (noun_dim+action_dim, seq_len)
                      - 前noun_dim行是noun特征
                      - 后action_dim行是action特征
            action_type: int，选择的ActionModel索引 (0 到 model_count-1)
        
        Returns:
            new_action_type: 形状 (action_dim,) - 预测的下一个action
        """
        # 验证action_type有效性
        if not isinstance(action_type, int):
            action_idx = int(action_type) - 1
        if action_idx < 0 or action_idx >= self.model_count:
            raise ValueError(f"action_type {action_type} 超出范围 [0, {self.model_count-1}]")
        
        # 将输入从 (noun+action, seq_len) 转换为 (1, seq_len, noun+action)
        model_input = input_2d.t().unsqueeze(0)  # (1, seq_len, noun_dim+action_dim)
        
        # 通过选中的ActionModel前向传播
        device = next(self.action_models[action_idx].parameters()).device
        model_input = model_input.to(device)
        new_action = self.action_models[action_idx](model_input)  # (1, action_dim)
        
        # 返回预测的新action，去掉batch维
        return new_action.squeeze(0)  # (action_dim,)
    
    def forward_with_sequence_build(self, input_2d: torch.Tensor, action_type: int, steps: int = 1) -> dict:
        """
        多步预测 - 逐步前向推导并构建新序列
        
        Args:
            input_2d: 初始输入 (noun_dim+action_dim, seq_len)
            action_type: 初始的ActionModel索引
            steps: 预测步数
        
        Returns:
            dict 包含：
                - 'final_sequence': 最终序列 (noun_dim+action_dim, seq_len+steps)
                - 'predictions': 所有预测的actions列表
                - 'action_types': 每一步使用的模型索引列表
        """
        current_seq = input_2d.clone()
        predictions = []
        action_types = []
        current_action_type = action_type
        
        for step in range(steps):
            # 前向传播得到预测的新action
            pred_action = self.forward(current_seq, current_action_type)  # (action_dim,)
            
            predictions.append(pred_action.detach().cpu())
            action_types.append(current_action_type)
            
            # 从最后一列提取noun特征
            last_col = current_seq[:, -1]
            noun_last = last_col[:self.noun_dim]
            
            # 使用(最后noun + 预测action)构造新列并拼接到序列
            new_col = torch.cat([noun_last, pred_action], dim=0).unsqueeze(1)
            current_seq = torch.cat([current_seq, new_col], dim=1)
            
            # 更新action_type为新预测的action的前model_count维的argmax
            if pred_action.numel() >= self.model_count:
                current_action_type = int(torch.argmax(pred_action[:self.model_count]).item())
        
        return {
            'final_sequence': current_seq,
            'predictions': predictions,
            'action_types': action_types
        }

