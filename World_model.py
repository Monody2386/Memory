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

