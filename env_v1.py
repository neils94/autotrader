from torchvision import transforms
import time
import numpy as np
import pyautogui
import requests
import json
import torch

token = 'eyJhbGciOiJSUzUxMiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI0ZmVjZjE4YmY5N2IwYWYwZjIyNGY1ZTAzYjhhNTcyNiIsInBlcm1pc3Npb25zIjpbXSwidG9rZW5JZCI6IjIwMjEwMjEzIiwiaW1wZXJzb25hdGVkIjpmYWxzZSwiaWF0IjoxNjczNTcxMzYzLCJyZWFsVXNlcklkIjoiNGZlY2YxOGJmOTdiMGFmMGYyMjRmNWUwM2I4YTU3MjYifQ.VBvrp1uW4GgFPBHzu5rx-izktAccznZAU6yp_uSt1QBMCm-N3_-DAt31OPqV83eg0SU4hxJ3fnnujDNONTXysgXl6RH734ilISRqE7FNMDpZseEpeu6wImwADmvJN1YxSEZM2kSeKAC7fPdeIY_JW2fAYRH2U7SFVOJbhDW7C5P348gYZpYpjCddzaTDjeyywbQw7uJdnnVZaM7dpOyhPJAbTjnAX9xQ5gq-sPMXpIUwdzY97F3sSEefxyVkPp70dK6fjQhsuut6oemNBHuVE-z8_7BSnCgjqw7Sedx5fDpjjaG-giFkTHhjDJtvT0DHv3FswMiZ5Flu-u61KczAnXnrmzXj0QJHwXcJuvL7YdOmtWP6YHa-P3ojF6ZtFsgs8pShDTctkV4GTH19e4fL7rWQTFDwWJt9sDc-ECePpW_1eFRuXIk_8FHhxcZbT_yhNOy8FlEoFEDSFu1tKKhSstvoFl_IbFkP1yt8tuwxgq7Zh5ztAS9ljFjiMvs-_CwVPNsEfVktmphz_R2QpVTMk58FFKSiqeYEuyGdSD-EzCvJMyD-tlTRPpjqygHn2GUqlRpAuS6FUO5a2iG9A7HYUdry1CSFLr_QT_Rqi_ZASgWx0AeNJbzjURIrEMk_rdHUCQlim__3LuM9KaIORo5ju8LzmWRzqXXD7n1Nco2OtDo'

class meta_env():
    def __init__(self):
        self.header = {"Content-Type": "application/json", "Accept": "application/json", "auth-token": "eyJhbGciOiJSUzUxMiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI0ZmVjZjE4YmY5N2IwYWYwZjIyNGY1ZTAzYjhhNTcyNiIsInBlcm1pc3Npb25zIjpbXSwidG9rZW5JZCI6IjIwMjEwMjEzIiwiaW1wZXJzb25hdGVkIjpmYWxzZSwiaWF0IjoxNjczNTcxMzYzLCJyZWFsVXNlcklkIjoiNGZlY2YxOGJmOTdiMGFmMGYyMjRmNWUwM2I4YTU3MjYifQ.VBvrp1uW4GgFPBHzu5rx-izktAccznZAU6yp_uSt1QBMCm-N3_-DAt31OPqV83eg0SU4hxJ3fnnujDNONTXysgXl6RH734ilISRqE7FNMDpZseEpeu6wImwADmvJN1YxSEZM2kSeKAC7fPdeIY_JW2fAYRH2U7SFVOJbhDW7C5P348gYZpYpjCddzaTDjeyywbQw7uJdnnVZaM7dpOyhPJAbTjnAX9xQ5gq-sPMXpIUwdzY97F3sSEefxyVkPp70dK6fjQhsuut6oemNBHuVE-z8_7BSnCgjqw7Sedx5fDpjjaG-giFkTHhjDJtvT0DHv3FswMiZ5Flu-u61KczAnXnrmzXj0QJHwXcJuvL7YdOmtWP6YHa-P3ojF6ZtFsgs8pShDTctkV4GTH19e4fL7rWQTFDwWJt9sDc-ECePpW_1eFRuXIk_8FHhxcZbT_yhNOy8FlEoFEDSFu1tKKhSstvoFl_IbFkP1yt8tuwxgq7Zh5ztAS9ljFjiMvs-_CwVPNsEfVktmphz_R2QpVTMk58FFKSiqeYEuyGdSD-EzCvJMyD-tlTRPpjqygHn2GUqlRpAuS6FUO5a2iG9A7HYUdry1CSFLr_QT_Rqi_ZASgWx0AeNJbzjURIrEMk_rdHUCQlim__3LuM9KaIORo5ju8LzmWRzqXXD7n1Nco2OtDo"}
        self.params =  {"accountId": "ff1d0714-5768-4ce1-ab38-d1aacfbc5c8b"}
    def account_info(self):
        acc_info  = requests.get('https://mt-client-api-v1.vint-hill.agiliumtrade.ai/users/current/accounts/ff1d0714-5768-4ce1-ab38-d1aacfbc5c8b/account-information', params = self.params,headers= self.header)
        #acc_info = acc_info.json()

        return acc_info
    
    def get_balance(self):
        acc_info  = requests.get('https://mt-client-api-v1.vint-hill.agiliumtrade.ai/users/current/accounts/ff1d0714-5768-4ce1-ab38-d1aacfbc5c8b/account-information', params =self.params,headers= self.header)
        balance = acc_info.json()['balance']


        return balance
    
    def time():
        time = requests.get('https://mt-client-api-v1.vint-hill.agiliumtrade.ai/users/current/accounts/865d3a4d-3803-486d-bdf3-a85679d9fad2/server-time')
        time = time.json()

    def enter_trade(self, symbol: str, volume: float, action):
        data = {
            "symbol": symbol,
            "actionType": action,
            "volume": volume,
            }
        data = json.dumps(data)
        enter_trade = requests.post("https://mt-client-api-v1.vint-hill.agiliumtrade.ai/users/current/accounts/ff1d0714-5768-4ce1-ab38-d1aacfbc5c8b/trade", headers=self.header, data = data)

        print(enter_trade.text)

    def get_open_trades(self):
        open_trades = requests.get("https://mt-client-api-v1.vint-hill.agiliumtrade.ai/users/current/accounts/ff1d0714-5768-4ce1-ab38-d1aacfbc5c8b/positions", params = self.params, headers = self.header)
        open_trades = open_trades.json()[0]
        
        return open_trades

    def close_position(self):
        position_id = self.get_open_trades()['id']
        data = {"actionType": "POSITION_CLOSE_ID", "positionId": position_id}
        data = json.dumps(data)
        close = requests.post("https://mt-client-api-v1.vint-hill.agiliumtrade.ai/users/current/accounts/ff1d0714-5768-4ce1-ab38-d1aacfbc5c8b/trade", headers=self.header, data= data)

        return close, position_id

    def trade_result(self, position_id):
        trade_result = requests.get('https://mt-client-api-v1.vint-hill.agiliumtrade.ai/users/current/accounts/ff1d0714-5768-4ce1-ab38-d1aacfbc5c8b/history-deals/position/'+position_id, params= self.params, headers=self.header)
        trade_result = trade_result.json()[1]['profit']

        return trade_result

    def acc(self, position_id):
        result = requests.get('https://mt-client-api-v1.vint-hill.agiliumtrade.ai/users/current/accounts/ff1d0714-5768-4ce1-ab38-d1aacfbc5c8b/history-orders/position/'+position_id,params= self.params, headers=self.header)
        result = result.json()

        return result

    def calculate_reward(self, trade_result, close:bool):
        trade_result = float(trade_result)
        balance = float(self.get_balance())
        if close == True:
            reward = (trade_result/balance) * 100
        else:
            open_trade = self.get_open_trades()['unrealizedProfit']
            reward = (open_trade/balance) * 100
        
        return reward

    def screenshot(self):
        
        screenshot_taken = pyautogui.screenshot()
        screenshot_taken = screenshot_taken.convert('RGB')
        return screenshot_taken

    def preprocess(self, screenshot):
        grayscale = transforms.functional.to_grayscale(screenshot)
        resized_img = transforms.functional.resize(grayscale, [84,135])
        cropped_img = transforms.functional.crop(resized_img, 0,0,84,84)
        state = transforms.functional.to_tensor(cropped_img).permute(0,2,1)
        return state

    def stacked_frames(self):
        stacks = []
        for i in range(4):
            state = self.screenshot()
            state = self.preprocess(state)
            stacks.append(state)

        state = torch.stack(stacks,dim=1)
        state = torch.squeeze(state)

        return state

    def step(self, action_probs, action):
        leverage = 500
        account_balance = float(self.get_balance())
        max_position_sizing =  account_balance * leverage
        lot_sizing= (max_position_sizing * 0.05 * float(action_probs))/int(100000)
        rewards_list = []
        temp_rewards = []
        action = int(action)

        if action == 0:
            self.enter_trade( "GBPUSD", lot_sizing, "ORDER_TYPE_BUY") 
        elif action == 1:
            self.enter_trade("GBPUSD", lot_sizing, "ORDER_TYPE_SELL")
        while ((action == 0) or (action==1)):
            time.sleep(60 - time.gmtime()[5] % 60)
            temp_reward = self.calculate_reward(trade_result=0.000, close=False)
            temp_rewards.append(temp_reward)
            if len(temp_rewards) >= 4:
                if temp_rewards[-3] - temp_rewards[-1] > 0:
                    close, position_id = self.close_position()
                    break
        trade_results = self.trade_result(position_id)
        reward = self.calculate_reward(trade_result=trade_results, close=True)
        rewards_list.append(reward)
        rewards = np.asarray(rewards_list, dtype=float, order='C')
        screenshot = self.screenshot()
        time.sleep(59)
        next_state = self.preprocess(screenshot)
        
        return rewards, next_state

