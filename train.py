from contextlib import nullcontext
from AdvantageAgent import AC_agent
from Environment import environment
from Model import Actor, Critic
import time


alpha = 0.99
beta = 0.01
eps= 0.9
gamma = 0.99
lamb = 1e-2
tau = 1e-3
state= nullcontext

lr = 1e-4
wd = 1e-4
buffer_size = int(1e5)
batch_size = 64

env = environment()

agent = AC_agent(batch_size, buffer_size, state, linear_input=2592, action_size=2, hidden_layer=256, hidden2_layer=256, input_channels=1, input_channels2=16, output_channels=16, output_channels2=32, 
    kernel_size=8, kernel_size2=4, stride=4, stride2=2)

def train_advantage(trading_week, trading_day):

    scores = []
    for i in range(trading_week):
        state = env.screenshot()
        if time.gmtime()[3] == 21:
            sleeptime = (60 % time.gmtime()[4]) * 60 + time.gmtime()[5]
            time.sleep(3600 - sleeptime % 3600) 

        for i in range(trading_day):
            action_probs, action = agent.act(state=state)
            reward, next_state, close = env.step(action_probs, action)
            assert close == 1
            agent.step(state=state, action=action, action_probs=action_probs, reward=reward, next_state=next_state, close=close)
            scores.append(reward)
            state = next_state
        



    return scores

scores = train_advantage(trading_week=1, trading_day=50)

print(scores)
