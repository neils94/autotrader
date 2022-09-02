from torchvision import transforms
import requests
import pyautogui
import json
import numpy as np
import time
import math
import numpy as np
import sys


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

#state, action, reward, next_state, open, close = experiences

#trade_result = retrieve the last single trade from API


class environment():
    """
    TO-DO LIST:
    - create a vectorized state space to store as 'state'
    - rewards will be distributed during open trades and after closing trades
    """
    """

    Properly working functions:
    -enter trade
    -close trade
    -get balance
    -open trades
    -get trade result

    Design Questions:
    - what is the trades exit condition?
    """

    def __init__(self, state, action: int):
        self.action = action
        self.state = state
        self.headers = {"Authorization": "Bearer 9d0b7ba6935a259276e4c4940e18c219-b2952fcc87ddd20980fc12b2c371786d", "Content-Type": "application/json", "Accept-Datetime-Format": "RFC3339"}

    def enter_trade(self, units: str, pair: str):
        #longs are in positive units and shorts are in negative units
        data = {
        "order": {
            "units": f"{units}",
            "instrument": pair,        
            "timeInForce": "FOK",
            "type": "MARKET",
            "positionFill": "DEFAULT"
            }
        }
        data = json.dumps(data)
        enter_trade = requests.post("https://api-fxpractice.oanda.com/v3/accounts/101-001-21526871-001/orders", headers=self.headers, data=data)

        return enter_trade

    def get_trade_result(self):
        params = {'state': 'CLOSED', 'count': int(1)}
        trade_result = requests.get('https://api-fxpractice.oanda.com/v3/accounts/101-001-21526871-001/trades', params=params, headers=self.headers)
        trade_result = trade_result.json()
        trade_result = trade_result["trades"][0]["realizedPL"]

        return trade_result

    def close_trade(self, trade_type: str):
        #longUnits or shortUnits depending one which open position
        data={trade_type: "ALL"}
        data = json.dumps(data)
        close = requests.put('https://api-fxpractice.oanda.com/v3/accounts/101-001-21526871-001/positions/USD_JPY/close', headers=self.headers, data=data)



        return close

    def get_balance(self):
        account_summary = requests.get("https://api-fxpractice.oanda.com/v3/accounts/101-001-21526871-001/summary", headers=self.headers)
        account = account_summary.json()
        balance = account['account']['balance']

        return balance

    def open_trades(self):
        #trade_type = long or short
        open_trades = requests.get("https://api-fxpractice.oanda.com/v3/accounts/101-001-21526871-001/openTrades", headers=self.headers)
        open_trades = open_trades.json()
        #detect if position is long or short
        pl = open_trades['trades'][0]['unrealizedPL']

        assert pl != 0.0000, f"Expected profit or loss, got {pl}"
        
        return pl

    def calculate_reward(self, trade_result: float, close:bool):
        trade_result = float(trade_result)
        balance = float(self.get_balance())
        if close == True:
            if trade_result > 0:
                reward = (trade_result/balance) * 10
            else:
                reward = (trade_result/balance) * 10
        else:
            open_trade = self.open_trades()
            reward = float(open_trade) * 1e-2
        
        return reward

    def screenshot(self):
        screenshot_taken = pyautogui.screenshot()
        return screenshot_taken

    def preprocess(self, screenshot):
        grayscale = transforms.functional.to_grayscale(screenshot)
        resized_img = transforms.functional.resize(grayscale, [84,135])
        cropped_img = transforms.functional.crop(resized_img, 0,0,84,84)
        state = transforms.functional.to_tensor(cropped_img).permute(0,2,1)
        return state

    def step(self, action_probs, action):
        leverage = 50
        account_balance = self.get_balance()
        account_balance = float(account_balance)
        max_position_sizing =  account_balance * leverage
        lot_sizing = math.floor(max_position_sizing * 0.05 * action_probs)
        rewards_list = []
        next_states = []
        closed = []
        temp_rewards = []
        reward = 0
        temp_reward = 0

        if action == 0:
            units = -lot_sizing
            self.enter_trade(units=units, pair="USD_JPY")
        elif action == 1:
            self.enter_trade(units=lot_sizing, pair="USD_JPY")
        while ((action == 0) or (action==1)):
            time.sleep(60 - time.gmtime()[5] % 58)
            temp_reward = self.calculate_reward(trade_result=0.000, close=False)
            temp_rewards.append(temp_reward)
            if len(temp_rewards) >= 3:
                if temp_rewards[-3] - temp_rewards[-1] > 0:
                    self.close_trade("shortUnits" if action == 0 else "longUnits")
                    break
        screenshot = self.screenshot() 
        next_state = self.preprocess(screenshot)
        next_states.append(next_state)
        close = 1
        closed.append(close)
        trade_result = self.get_trade_result()
        reward = self.calculate_reward(trade_result=trade_result, close=True)
        rewards_list.append(reward)
        rewards = np.array(rewards_list, order='C')
        
        return rewards, next_states, closed


env = environment(state=1,action=1)

rewards, next_states, closed = env.step(action_probs=0.73, action=1)
