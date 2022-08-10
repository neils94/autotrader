import numpy as np
import torch 
import torchvision
from torch import nn
from torchvision import transforms
import requests
import pyautogui


state, action, reward, next_state, open, close = experiences

trade_result = #retrieve the last single trade from API

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
    """

    def __init__(self, state, action: int, closed: bool):
        self.action = action
        self.state = state
        self.closed = closed
        self.headers = {"Authorization": "Bearer 9d0b7ba6935a259276e4c4940e18c219-b2952fcc87ddd20980fc12b2c371786d", "Content-Type": "application/json", "Accept-Datetime-Format": "RFC3339"}

    def enter(self, units):
        #longs are in positive units and shorts are in negative units
        data = {
        "order": {
            "units": units,
            "instrument": "USD_JPY",        
            "timeInForce": "FOK",
            "type": "MARKET",
            "positionFill": "DEFAULT"
            }
        }
        data = json.dumps(data)
        open_trade = requests.post("https://api-fxpractice.oanda.com/v3/accounts/101-001-21526871-001/orders", headers=self.headers, data=data)

        return open_trade

    def get_trade_result(self):
        trade_result = requests.get('https://api-fxpractice.oanda.com/v3/accounts/101-001-21526871-001/trades', data={'state': 'CLOSED', 'count': '1'})
        return trade_result

    def close(self):
        data={
        "close": {
            "longUnits": "ALL", 
            "shortUnits": "ALL"
            }
        }

        data = json.dumps(data)
        close_trade = requests.put('https://api-fxpractice.oanda.com/v3/accounts/101-001-21526871-001/positions/USD_JPY/close', headers=self.headers)

    def get_balance(self):
        account_summary = requests.get("https://api-fxpractice.oanda.com/v3/accounts/101-001-21526871-001/summary", headers=self.headers)

        return account_summary




        def calculate_reward(self, trade_result, balance):

            current_balance = trade_result + balance

            if current_balance > balance:
                
                reward = torch.log(trade_result)
            else:
                reward = -torch.log(trade_result)

            return reward


    #def enter(self, units: string):
        #post_trade = requests.post('https://api-fxtrade.oanda.com/v3/accounts/{001-001-7438950-001}/orders', data={"units": units, "instrument":"USD_JPY", "timeInForce":"FOK","type":"MARKET","positionalFill": "DEFAULT"})

    #def get_trade_result(self):
        #return trade_result = requests.get('https://api-fxtrade.oanda.com/v3/accounts/{001-001-7438950-001}/trades', data={'state': 'CLOSED', 'count': '1'})

    #def get_account_balance(self):
        #return balance = requests.get('https://api-fxtrade.oanda.com/v3/accounts/{001-001-7438950-001}/orders', data=)




    
    



    

    

    

    