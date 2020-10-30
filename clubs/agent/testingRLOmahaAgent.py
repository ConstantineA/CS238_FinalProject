import random

from . import base
from collections import defaultdict
import math


class TestingRLOmahaAgent(base.BaseAgent):
    def __init__(self):
        super().__init__()

    
    #Actions are one of three things:
    #0) Call/check
    #1) Fold
    #2) Max Raise
    #Thus, there are only three actions that can be done.

    alphaLearningRate = 0.2
    discount = 1
    weights = defaultdict(float)
    explorationProb = 0.1

    totalRewards = 0

    currBestAction = 0 #assume starting best action is to call/check
    oldState = []
    
    def getActions(self,obs):
        #all of the possible actions include folding, calling, or raising
        actions = [a for a in range(obs["call"], obs["call"] + obs["max_raise"], 1)]
        actions.append(-1)
        return actions

    def getState(self,obs):
        featuresList = []
        print("pot",obs["pot"])
        featuresList.extend(obs["community_cards"])
        featuresList.extend(obs["hole_cards"])
        print(featuresList)
        featuresList.append(obs["pot"])
        print("pot",obs["pot"])
        print(featuresList)

        return featuresList

    def getQValue(self, state, a):
        #the tuple (state,a) is used as the key for the defaultdictionary
        print("state:", state)
        print("action:", a)
        state.append(a)
        print("state with action:", state)
        key = tuple(state)
        print("key: ", key)

        print("weights:", self.weights)
        return self.weights[key]


    def chooseAction(self, state, actions):
        if random.random() < self.explorationProb:
            return random.choice(actions)
        else:
            maxAction = 0 #default is to check/call 
            maxQ = -math.inf
            for a in actions:
                currVal = self.getQValue(state, a)
                if currVal > maxQ:
                    maxQ = currVal
                    maxAction = a
            return maxAction

    def forwardRL(self,obs):
        actions = self.getActions(obs)
        state = self.getState(obs)
        self.oldState = state

        self.currBestAction = self.chooseAction(state,actions)
        return self.currBestAction


    def update(state, reward, sPrime, actions):
        newQ = 0
        for a in actions:
            q = self.getQ(sPrime, a)
            if q > newQ:
                newQ = q
        #TAKE A LOOK AT THIS ALGORITHM, MIGHT BE WRONG
        updateValue = self.alphaLearningRate * (reward + self.discount*(newQ - self.getQ(state,self.currBestAction)))
        self.weights[self.getState(state,self.currBestAction)] += updateValue

    def setRewardandUpdate(self, obs, reward):
        print("weights:",self.weights)
        sPrime = self.getState(obs)
        self.totalRewards += reward
        actions = self.getActions(obs)
        self.update(self.oldState, reward, sPrime, actions)

    '''
    def getActions(self,obs):
        #all of the possible actions include folding, calling, or raising
        #for a in range(obs["call"], obs["call"] + obs["max_raise"], 1):
            #print(a)
        actions = [a for a in range(obs["call"], obs["call"] + obs["max_raise"], 1)]
        
        actions.append(-1)
        #print(actions)
        #print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        return actions

    def getFeatures(self,obs):
        featuresList = []
        print(obs)
        featuresList.append(obs["community_cards"])
        featuresList.append(obs["hole_cards"])
        featuresList.append(obs["pot"])

        return featuresList

    def forwardRL(self,obs):
        features = self.getFeatures(obs)
        possibleActions = self.getActions(obs)
        return -1
    '''

   

    def reverseRL(self,obs):
        return -1

    
    def random(self,obs):
        #TODO
        #later on, maybe create an array with all of the
        #possible bet sizes + -1 for fold
        #betAction = [for i in range()]
        action = random.random()
        if action < 0.3: #fold
            return -1
        elif action < 0.6: # bet/call
            return obs["call"]

    def alwaysFold(self):
        return -1

    def alwaysBetCall(self,obs):
        return obs["call"]

    def alwaysMaxRaise(self,obs):
        return obs["max_raise"]


    def sayHello(self, i):
        print("hello world, i'm agent number:", i)

    def act(self, obs):
        '''
        if obs["action"] == 0:

            return 5

        return 5
        '''
        return self.alwaysBetCall(obs)

        
        #if obs["action"] == 0: #player 1
        #    return self.forwardRL(obs)
        #return self.reverseRL(obs)
        