import random

class ReplayMemory():
    def __init__(self, max_size=-1):
        self.memory = []
        self.max_size = max_size # -1 means infinite buffer size
    
    def push(self, *args):
        if (self.max_size > 0 and len(self.memory) > self.max_size):
            self.memory.pop(0)
        self.memory.append(args)
    
    def sample(self, batch_size):
        bs = len(self.memory) if batch_size < 0 or batch_size > len(self.memory) else batch_size
        return random.sample(self.memory, k=bs)