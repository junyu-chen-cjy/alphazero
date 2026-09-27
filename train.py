from torch.utils.data import DataLoader
from data.dataset import GomokuDataset
from collections import deque

buffer = deque(maxlen=10000)
train_dataset = GomokuDataset(buffer)
   
train_loader = DataLoader(
    dataset = train_dataset,
    batch_size = 32,
    shuffle = True,
    drop_last = True,
    pin_memory = True
)