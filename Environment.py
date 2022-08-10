import numpy as np
import torch 
import torchvision
from torch import nn
from torchvision import transforms
import requests


state, action, reward, next_state, trade = experiences

trade_result = #retrieve the last single trade from API


class environment():
    """
    -API (string): API that is being called i.e: Oanda, Crypto/stocks
    """

    def __init__(self, state, action):
        self.action = action
        self.state = state

    def screenshot():
        screenshot_taken = pyautogui.screenshot()
        return screenshot_taken

    def preprocess(screenshot):
        grayscale = transforms.functional.to_grayscale(screenshot)
        resized_img = transforms.functional.resize(grayscale, [84,135])
        cropped_img = transforms.functional.crop(resized_img, 0,0,84,84)
        tensor_img = transforms.functional.to_tensor(cropped_img).permute(0,2,1)
        return tensor_img




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
        return trade_result = requests.get('https://api-fxtrade.oanda.com/v3/accounts/{001-001-7438950-001}/trades', data={'state': 'CLOSED', 'count': '1'})

    #def get_account_balance(self):
        #return balance = requests.get('https://api-fxtrade.oanda.com/v3/accounts/{001-001-7438950-001}/orders', data=)


    def reward(self):

        #give reward to agent so it can be stored
        if done:

            reward = Environment.calculate_reward()



    
    



    

    

    

    