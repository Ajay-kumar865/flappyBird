from collections import deque
import random
class ReplayMemory():
    def __init__(self,maxlen,seed=None):
        self.memory=deque([],maxlen=maxlen)
        self.rng = random.Random(seed)
    def append(self,new_exp):
        self.memory.append(new_exp)
    def sample(self,sample_size):
        return self.rng.sample(self.memory,sample_size)
    def __len__(self):
         return(len(self.memory))
 
