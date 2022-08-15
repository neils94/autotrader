import torch 
from torchvision import transforms
import requests
import pyautogui
import json


#state, action, reward, next_state, open, close = experiences

#trade_result = retrieve the last single trade from API

def screenshot():
    screenshot_taken = pyautogui.screenshot()
    return screenshot_taken

def preprocess(screenshot):
    grayscale = transforms.functional.to_grayscale(screenshot)
    resized_img = transforms.functional.resize(grayscale, [84,135])
    cropped_img = transforms.functional.crop(resized_img, 0,0,84,84)
    tensor_img = transforms.functional.to_tensor(cropped_img).permute(0,2,1)
    return tensor_img


class environment():
    """
    TO-DO LIST:
    - create a vectorized state space to store as 'state'
    - rewards will be distributed during open trades and after closing trades
    - return account balance every minute; 
    """
    """
    Total Mapping:
    -Screenshot
    -Pre-process screenshot
    -Input to actor model
    -Actor takes action
    -Rewards are dispersed during open trades and immediately after closing
    -Update actor and critic networks with rewards

    Properly working functions:
    -enter trade
    -close trade
    -get balance
    -open trades
    -get trade result
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
        open_trade = requests.post("https://api-fxpractice.oanda.com/v3/accounts/101-001-21526871-001/orders", headers=self.headers, data=data)

        return open_trade

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

    def open_trades(self, trade_type: str):
        #trade_type = long or short
        open_trades = requests.get("https://api-fxpractice.oanda.com/v3/accounts/101-001-21526871-001/openPositions", headers=self.headers)
        open_trades = open_trades.json()
        #detect if position is long or short
        units = open_trades['positions'][0][trade_type]['pl']
        
        return units

    def calculate_reward(self, trade_result, open, close, trade_type):
        if close == True:
            if trade_result > 0:
                reward = torch.log(trade_result)
            else:
                reward = -(torch.log(trade_result))
        elif open and close == False:
            reward = 0
        elif open == True:
            open_trade = environment.open_trades(trade_type)
            reward = open_trade * 1e-6
            #set break conditions  

        return reward




    
    



    

    

    

    
