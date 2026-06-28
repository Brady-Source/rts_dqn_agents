import random
from collections import deque
import numpy as np

# Establishing the Experience Replay Buffer

class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        state = np.stack(state)
        next_state = np.stack(next_state)
        action = np.array(action, dtype=np.int64)
        reward = np.array(reward, dtype=np.float32)
        done = np.array(done, dtype=np.float32)
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)