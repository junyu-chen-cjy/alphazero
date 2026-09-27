import torch 
import numpy as np
from torch.utils.data import Dataset

class BoardTransform():
    def __call__(self, board):
        #棋盘数据转双通道张量
        ch_self = (board.board == board.current_player).astype(np.float32)
        ch_opp = (board.board == -board.current_player).astype(np.float32)
        x = torch.from_numpy(np.stack([ch_self,ch_opp], axis = 0))
        return x
    
class GomokuDataset(Dataset):
    def __init__(self,buffer):
        #存buffer
        self.buffer = buffer
        self.transform = BoardTransform()
        
    def __len__(self):
        return len(self.buffer)
    
    def __getitem__(self,idx):
        #拆buffer
        exp = self.buffer[idx]
        board_tensor = self.transform(exp.board)
        policy_tensor = torch.tensor(exp.policy,dtype = torch.float32)
        value_tensor = torch.tensor(exp.value,dtype = torch.float32)
        return (board_tensor,value_tensor,policy_tensor)


        