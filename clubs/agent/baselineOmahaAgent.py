import random

from . import base


class BaselineOmahaAgent(base.BaseAgent):
    def __init__(self):
        super().__init__()

    #BASELINE METHODS
    #Assume no checking
    #Random
    #Max Bet
    #Always Fold
    #Always Call

    #Three moves: Bet/Call, Fold, Raise


    
    def randomAction(self,obs):
        #TODO
        #later on, maybe create an array with all of the
        #possible bet sizes + -1 for fold
        #betAction = [for i in range()]
        action = random.random()
        if action < 0.1: #fold
            return -1
        elif action < 0.6: # bet/call
            return obs["call"]
        else:
            return obs["max_raise"]

    def alwaysFold(self):
        return -1

    def alwaysBetCall(self,obs):
        return obs["call"]

    def alwaysMaxRaise(self,obs):
        return obs["max_raise"]




    def act(self, obs):
        #testing shit
        '''
        if obs["action"] == 0:

            return 5

        return 5
        
        if obs["action"] == 0: #player 1
            return self.alwaysBetCall(obs)
        return self.alwaysBetCall(obs)
        '''
        randAction = self.randomAction(obs)
        print(randAction)
        return randAction
