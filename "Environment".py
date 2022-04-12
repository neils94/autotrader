"Environment"

import numpy as np
import torch 
import tensorflow as tf 
import fxcmpy

state, action, reward, next_state, trade = experiences

trade_result = #retrieve the last single trade from API


class Environment():


    def __init__(self):
        #initialize global variables that you will need throughout the class

        self.balance = balance #collect balance from fetching api
        self.trade_result = trade_result #from api fetch trade result
        self.current_batch = pass





    def calculate_reward(self, trade_result, balance):

        current_balance = trade_result + balance

        ##find a reward function that works by scaling reward based on size of profits/losses
        if current_balance > balance:

            reward = tf.float32(tf.log(trade_result)).type(float)
        else:
            reward = -tf.float32(balance+tf.log(trade_result)).type(float)

        return reward


    def reward(self):

        #give reward to agent so it can be stored
        if done:

            reward = Environment.calculate_reward()





    def step(self, state, action):

        #be able to take state as input of pictures
        #take action as input and send action to the api environment
        #collect reward from rewards functions


        state = tf.unsqueeze(state)


        stop_loss = tf.min(#previous candle minimum)

        stop_loss = tf.max(#previous candle maximum)


    def memory(self):

        #send all images to cloud storage
    
    



    

    

    

    