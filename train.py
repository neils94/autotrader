from contextlib import nullcontext
from AdvantageAgent import AC_agent
from meta_env2 import meta_env2
import time
import pyautogui
import torch
import sys
import wandb
import requests
import Agent2

eps= 0.9
gamma = 0.99
lamb = 1e-2
tau = 1e-3
state= nullcontext

lr = 1e-4
wd = 1e-4
buffer_size = int(1e5)
batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def writelines(self, datas):
       self.stream.writelines(datas)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)


sys.stdout = Unbuffered(sys.stdout)

agent = AC_agent(batch_size, buffer_size, state, linear_input=2592, action_size=2, hidden_layer=256, hidden2_layer=256, input_channels=4, input_channels2=16, output_channels=16, output_channels2=32, 
    kernel_size=8, kernel_size2=4, stride=4, stride2=2)
env = meta_env2()

agent2 = Agent2(batch_size, buffer_size, state, linear_input=2592, action_size=2, hidden_layer=256, hidden2_layer=256, input_channels=4, input_channels2=16, output_channels=16, output_channels2=32, 
    kernel_size=8, kernel_size2=4, stride=4, stride2=2)
def forward_advantage():
    closed = []
    scores = []
    state = env.stacked_frames()
    status_code = int(200)
    while ((env.account_info()).status_code == status_code):
            time.sleep(60 - time.gmtime()[5] % 60)
            action_combo = agent.greedy_action(state)
            action_probs, action_1 = action_combo
            reward, action_2, next_state_1, next_state_2 = env.step(action_probs, action_1)
            assert action_2 == 1
            close = True
            closed.append(close)
            agent.memory_update(state, action_1, action_2, action_probs, reward, next_state_1, next_state_2, close)
            state = next_state_2
            scores.append(reward)
            pyautogui.press('shift')

    else:
        print(env.account_info().status_code)

    return scores
