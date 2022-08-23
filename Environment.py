from asyncio.windows_events import NULL
import torch 
from torchvision import transforms
import requests
import pyautogui
import json
import numpy as np
import time


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
            "units": units,
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
    
    def open_close_info(self):
        open_trades = requests.get("https://api-fxtrade.oanda.com/v3/accounts/101-001-21526871-001/openTrades", headers=self.headers)
        open_trades = open_trades.json()

        if open_trades["trades"][0]:
            closed = False

        else:
            closed = True
        
        return closed



    def open_trades(self, trade_type: str):
        #trade_type = long or short
        open_trades = requests.get("https://api-fxpractice.oanda.com/v3/accounts/101-001-21526871-001/openPositions", headers=self.headers)
        open_trades = open_trades.json()
        #detect if position is long or short
        pl = open_trades['positions'][0][trade_type]['pl']
        
        return pl

    def calculate_reward(self, trade_result, close, trade_type):
        if close == True:
            if trade_result > 0:
                reward = torch.log(trade_result)
            else:
                reward = -(torch.log(trade_result))
        else:
            open_trade = environment.open_trades(trade_type)
            reward = open_trade * 1e-6
            #set break conditions  

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
        account_balance = environment.get_balance()
        max_position_sizing =  (account_balance * leverage)
        units  = (max_position_sizing * 0.05) * action_probs 
        rewards = []
        reward = 0
        next_states = []

        if action == 0:
            units = -(units)
            units = str(units)
            entry = environment.enter_trade(units, pair="USD_JPY")
            open_trade = environment.open_trades('short')
            if open_trade > 0:
                while True:
                    time.sleep(60 - time.gmtime()[5] % 60)
                    screenshot = environment.screenshot() 
                    next_state = environment.preprocess(screenshot)
                    reward += (open_trade * 1e-2)
                    rewards.append(reward)
                    next_states.append(next_state)
                    break;
            else:
                reward -= (open_trade * 1e-2)
                rewards.append(reward)



        else:
            units = units
            units = str(units)
            entry = environment.enter_trade(units, pair="USD_JPY")
            while True:
                    screenshot = environment.screenshot() 
                    next_state = environment.preprocess(screenshot)
                    open_trade = environment.open_trades('long')

env = environment(state=1,action=1)
