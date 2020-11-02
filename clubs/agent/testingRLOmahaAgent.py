import random

from . import base
from collections import defaultdict
import math
import numpy as np


STR_RANKS = "23456789TJQKA"
INT_RANKS = list(range(13))
RANK_DICT = dict(zip(list(STR_RANKS), INT_RANKS))

class TestingRLOmahaAgent(base.BaseAgent):
    def __init__(self):
        super().__init__()

    #Gradient QLearning
    #actionSpace = [0,1,-1] #0 = call/check, #1 = max_raise, #-1 = fold
    gammaGlobal = 1
    gradientQ = 1
    thetaBetCheck = np.ones(119)
    thetaMaxRaise = np.ones(119)
    thetaFold = np.ones(119)
    alphaLearningRate = 0.2

    thetas = np.array([thetaBetCheck, thetaMaxRaise, thetaFold])
    previousStateFeatures = 1/13*np.ones(119)
    previousStateFeatures[0] = 0
    previousStateFeatures[1] = 200
    totalRewards = 0
    currBestAction = 0 #initialize our first action to bet/check

    def getActions(self,obs):
        #[call/check, max_raise, fold]
        #IMPLEMENT THIS LATER FOR COMPLEXITY
        actions = []
        actions.append(obs["call"])
        actions.append(obs["max_raise"])
        actions.append(-1)

        actionsNew = [0,1,2] #[bet/check, max_raise, fold]
        return actionsNew

#MAKE THIS FUNCTION IN CARD.PY
    def get_card_number(self,card):
	    rank_str = card.__str__()[0]
	    return RANK_DICT[rank_str]


#0 = 2; 1 = 3... pos 8 = 10, pos 9 = J, 10 = Q, 11 = K, 12 = Ace
    def getFeatures(self, obs):
        
        features = []
        #features.append(obs["pot"])
        features.append(1)
        playerNumber = obs["action"]
        features.append(obs["stacks"][playerNumber])

        #print("we've appended the pot, and stack")

        #OUR FOUR HOLE CARDS
        #First 4 vector features are for each hole card
        if (not obs["hole_cards"]):
            #print("no hole cards")
            #don't have hole cards, we're completely blind
            for i in range(52):
                features.append(1/13)
        else:
            #print("we have hole cards")
            for card in obs["hole_cards"]:
                counter = 0
                cardRank = self.get_card_number(card)
                while counter < 13:
                    counter += 1
                    if counter == cardRank:
                        features.append(1)
                    else:
                        features.append(0)

        #OUR FIVE COMMUNITY CARDS
        #print("setting community card features")
        communityCounter = 0
        for card in obs["community_cards"]:
            #print(card)
            communityCounter += 1

            counter = 0
            cardRank = self.get_card_number(card)
            while counter < 13:
                counter += 1
                if counter == cardRank:
                    features.append(1)
                else:
                    features.append(0)
            
        while communityCounter < 5:
            communityCounter += 1
            for i in range(13):
                #This is where we incorporate possible probability changes based on what we've seen
                #For now, ignore what we know from other hands
                features.append(1/13)
        
        #print(len(features))
        return np.array(features)
        
    def getOldQ(self):
        prevAction = self.currBestAction #this is the previous action taken
        prevStateFeatures = self.previousStateFeatures
        theta = self.thetas[prevAction]
        
        #print("LOOK HERE------------------------------")
        #print("self.thetas", self.thetas)
        #print("slef.thetas[0]", self.thetas[0])
        #print(theta)
        #print(prevAction)

        qVal = np.dot(theta,prevStateFeatures)
        #print("the dot product: ")
        #print("qval shape:", qVal.shape)
        #print(theta.shape, prevStateFeatures.shape)
        #print(self.thetas.shape)
        return qVal

    def gradFxn(self):
        return self.previousStateFeatures



    def update(self,a,r, obs):

        #print("WE ARE UPDATing")
        #return 5
        #the observation is for sprime
        
        #actionSpace = self.getActions(obs) #maybe instead of using the actual number for action space....have arbitrary
        #action values that correspond to check/bet, min_raise, min_raise +2, min_raise +4, ...max_raise, fold
        #actionSpace = [0,1,2] #[bet/check, max_raise, fold]
        gamma = self.gammaGlobal
        featuresVector = self.getFeatures(obs)
        alpha = self.alphaLearningRate

        u = -math.inf
        for a in self.getActions(obs):
            currTheta = self.thetas[a]
            currQValue = np.dot(currTheta, featuresVector)
            if currQValue > u:
                u = currQValue
        
        #print("reward:",r)
        #print("gamma:",gamma)
        #print("u:", u)
        #print("old q value:",self.getOldQ())
        left = (r + gamma * u - self.getOldQ())
        #print("left",left)
        #print(type(left))
        #print("---------------------------")
        #print(self.gradFxn())
        #print(type(self.gradFxn()))
        #print(left * self.gradFxn())
        delta = left * self.gradFxn()
        #delta = (r + gamma * u - self.getOldQ())*self.gradFxn()

        #Our scaling for the gradient is so that all elements add to one
        #ASK CHRIST ABOUT SCALING GRADIENT!!

        toMultiply = np.min([1/np.linalg.norm(delta),1])*delta
        self.thetas[a] += alpha * toMultiply
    

    def chooseAction(self,stateFeatures, actions):
        #INCORPORATE EPSIOLON GREEDY PROBABILITY; code is below
        maxQ = -math.inf
        maxAction = 0 #corresponds to nothing
        for a in actions:
            print(a)
            currTheta = self.thetas[a]
            print("currTheta",currTheta)
            currQValue = np.dot(currTheta, stateFeatures)
            print("currQvalue:",currQValue)
            if currQValue > maxQ:
                maxQ = currQValue
                maxAction = a
        return maxAction

    def forwardRL(self,obs):
        #print("in forwardRL")
        actions = self.getActions(obs)
        print(actions)
        #print("past getActions")
        currentStateFeatures = self.getFeatures(obs)
        print(currentStateFeatures)
        #print("past get Features")
        self.previousStateFeatures = currentStateFeatures
        self.currBestAction = self.chooseAction(currentStateFeatures,actions)
        
        #print("finished getting best action in forwardRL")
        print("current best action:", self.currBestAction)
        return self.currBestAction

    def setRewardandUpdate(self, obs, reward):
        #print("weights:",self.weights)
        nextStateFeatures = self.getFeatures(obs)
        self.totalRewards += reward
        actions = self.getActions(obs)
        #print("oldstate",self.oldState)
        #print("reward",reward)
        #print("sprime",sPrime)
        #print("actions",actions)

        self.update(self.currBestAction, reward, obs)

    def alwaysFold(self):
        return -1

    def alwaysBetCall(self,obs):
        return obs["call"]

    def alwaysMaxRaise(self,obs):
        return obs["max_raise"]

    def act(self, obs):
        
        if obs["action"] == 0: #player 1
            return self.forwardRL(obs)
        else:
            return self.alwaysBetCall(obs)

'''
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
        #print("pot",obs["pot"])
        featuresList.extend(obs["community_cards"])
        featuresList.extend(obs["hole_cards"])
        #print(featuresList)
        featuresList.append(obs["pot"])
        #print("pot",obs["pot"])
        #print(featuresList)

        return featuresList

    def getQValue(self, state, a):
        #the tuple (state,a) is used as the key for the defaultdictionary
        #print("state:", state)
        #print("action:", a)
        state.append(a)
        #print("state with action:", state)
        key = tuple(state)
        #print("key: ", key)

        #print("weights:", self.weights)
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


    def update(self,state, reward, sPrime, actions, obs):
        newQ = 0
        for a in actions:
            q = self.getQValue(sPrime, a)
            if q > newQ:
                newQ = q
        #TAKE A LOOK AT THIS ALGORITHM, MIGHT BE WRONG
        updateValue = self.alphaLearningRate * (reward + self.discount*(newQ - self.getQValue(state,self.currBestAction)))
        #self.weights[self.getState(state,self.currBestAction)] += updateValue
        self.weights[tuple(self.getState(obs))] += updateValue

    def setRewardandUpdate(self, obs, reward):
        #print("weights:",self.weights)
        sPrime = self.getState(obs)
        self.totalRewards += reward
        actions = self.getActions(obs)
        #print("oldstate",self.oldState)
        #print("reward",reward)
        #print("sprime",sPrime)
        #print("actions",actions)
        self.update(self.oldState, reward, sPrime, actions, obs)
  
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
        
        
        #if obs["action"] == 0:
         #   return self.alwaysMaxRaise(obs)
        #else:
         #   return self.alwaysBetCall(obs)


        
        if obs["action"] == 0: #player 1
            return self.forwardRL(obs)
        else:
            return self.alwaysBetCall(obs)

'''        