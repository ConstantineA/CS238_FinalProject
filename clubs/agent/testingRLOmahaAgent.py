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
    thetaBetCheck = 1/119 * np.ones(119)
    thetaMaxRaise = 1/119 * np.ones(119)
    thetaFold = 1/119 * np.ones(119)
    #thetaBetCheck = np.ones(119)
    #thetaMaxRaise = np.ones(119)
    #thetaFold = np.ones(119)
    alphaLearningRate = 0.2
    training = True


    thetas = np.array([thetaBetCheck, thetaMaxRaise, thetaFold])
    previousStateFeatures = 1/13*np.ones(119)
    previousStateFeatures[0] = 0
    previousStateFeatures[1] = 200
    totalRewards = 0
    currBestAction = 0 #initialize our first action to bet/check
    explorationProb = 0.7

    def getThetas(self):
        return self.thetas

    def setStartingThetas(self, givenThetas):
        self.thetas = givenThetas

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
        features.append(obs["pot"])
        #features.append(1)
        playerNumber = obs["action"]
        features.append(obs["stacks"][playerNumber])
        #print("player number:", playerNumber)
        #print("hole cards in getfeatures fxn",obs["hole_cards"])

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
                #print(card)
                #print(cardRank)
                #print("******")
                while counter < 13:
                    if counter == cardRank:
                        features.append(1)
                    else:
                        features.append(0)
                    counter += 1

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
        #print("features Vector in update fxn:",featuresVector)
        alpha = self.alphaLearningRate

        maxAction = -1
        u = -math.inf
        #print("looping through the best Q value for all actions")
        for a in self.getActions(obs):
            #print(a)
            currTheta = self.thetas[a]
            #print(currTheta)
            currQValue = np.dot(currTheta, featuresVector)
            #print(currQValue)
            if currQValue > u:
                u = currQValue
                maxAction = a
                #print(u)
        
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

        if np.any(delta):
            toMultiply = np.min([1/np.linalg.norm(delta),1])*delta
        else:
            toMultiply = 1
        self.thetas[maxAction] += alpha * toMultiply
    

    def chooseAction(self,stateFeatures, actions):
        #INCORPORATE EPSIOLON GREEDY PROBABILITY; code is below
        if random.random() < self.explorationProb:
            return random.choice(actions)
        else:
            maxQ = -math.inf
            maxAction = 0 #corresponds to nothing
            for a in actions:
                #print(a)
                currTheta = self.thetas[a]
                #print("currTheta",currTheta)
                currQValue = np.dot(currTheta, stateFeatures)
                #print("currQvalue:",currQValue)
                if currQValue > maxQ:
                    maxQ = currQValue
                    maxAction = a
            return maxAction

    def forwardRL(self,obs):
        #print(self.thetas[0])
        #print(self.thetas[1])
        #print(self.thetas[2])
        
        #print(obs)
        #print("in forwardRL")
        actions = self.getActions(obs)
        #print(actions)
        #print("past getActions")
        currentStateFeatures = self.getFeatures(obs)
        #print(currentStateFeatures)
        #print("past get Features")
        self.previousStateFeatures = currentStateFeatures
        self.currBestAction = self.chooseAction(currentStateFeatures,actions)
        
        #print("finished getting best action in forwardRL")
        #print("current best action:", self.currBestAction)

        actionForSimulator = -2
        if self.currBestAction == 0: #check/calling
            actionForSimulator = obs["call"]
        elif self.currBestAction == 1: #max raise
            actionForSimulator = obs["max_raise"]
        else: #fold
            actionForSimulator = -1 
        return actionForSimulator

    def setRewardandUpdate(self, obs, reward):
        #print("weights:",self.weights)
        nextStateFeatures = self.getFeatures(obs)
        self.totalRewards += reward
        actions = self.getActions(obs)
        #print("oldstate",self.oldState)
        #print("reward",reward)
        #print("sprime",sPrime)
        #print("actions",actions)
        #print("IN THE REWARD UPDATE FUNCTION------------------",self.currBestAction)

        self.update(self.currBestAction, reward, obs)

    def alwaysFold(self):
        return -1

    def alwaysBetCall(self,obs):
        return obs["call"]

    def alwaysMaxRaise(self,obs):
        return obs["max_raise"]

    def chooseTrainedAction(self, stateFeatures, actions):
        maxQ = -math.inf
        maxAction = -1 #corresponds to nothing
        for a in actions:
            currTheta = self.thetas[a]
            currQValue = np.dot(currTheta, stateFeatures)
            if currQValue > maxQ:
                maxQ = currQValue
                maxAction = a
        return maxAction

    def bestTrainedAction(self, obs):
        actions = self.getActions(obs)
        currentStateFeatures = self.getFeatures(obs)
        currBestAction = self.chooseTrainedAction(currentStateFeatures,actions)
        
        #print("finished getting best action in forwardRL")
        #print("current best action:", self.currBestAction)

        actionForSimulator = -2
        if currBestAction == 0: #check/calling
            actionForSimulator = obs["call"]
        elif currBestAction == 1: #max raise
            actionForSimulator = obs["max_raise"]
        else: #fold
            actionForSimulator = -1 
        return actionForSimulator

    def setTrainingOff(self):
        self.training = False

    def randomAction(self,obs):
        action = random.random()
        if action < 0.3: #fold
            return -1
        elif action < 0.6: # bet/call
            return obs["call"]
        else:
            return obs["max_raise"]

    top30HandsList = [ ['A','A','K','K'], ['A','A','J','T'], ['A','A','Q','Q'], ['A','A','J','J'], ['A','A','T','T'],
                   ['A','A','9','9'], ['J','T','9','8'],
                   ['K','K','Q','Q'], ['K','K','J','J'], ['K','Q','J','T'], ['K','K','T','T'], ['K','K','A','Q'],
                   ['K','K','A','J'], ['K','K','A','T'], ['K','K','Q','J'], ['K','K','Q','T'], ['K','K','J','T'],
                   ['Q','Q','J','J'], ['Q','Q','T','T'], ['Q','Q','A','K'], ['Q','Q','A','J'], ['Q','Q','A','T'],
                   ['Q','Q','K','J'], ['Q','Q','K','T'], ['Q','Q','J','T'], ['Q','Q','J','9'], ['Q','Q','9','9'],
                   ['J','J','T','T'], ['J','J','T','9']
                 ]

    high_face = ["T", "J", "Q", "K", "A"]
    face = ["2","3","4","5","6","7","8","9", "T","J","Q","K","A"]

    for i in range(len(face)):
        for j in range(len(face)):
            top30HandsList.append((["A", "A", face[i], face[j]]))

    akxxs = []
    for i in range(len(high_face)):
        for j in range(len(face)):
                akxxs.append((["A", "K", high_face[i], face[j]]))

    highFourInARowList = [ ['5','6','7','8'], ['6','7','8','9'], ['7','8','9','T'], ['8','9','T','J'], ['9','T','J','Q'],
                    ['T','J','Q','K'], ['J','Q','K','A']
                    ]

    lowFourInARowList = [ ['4','5','6','7'], ['5','6','7','8'], ['6','7','8','9'], ['7','8','9','T'], ['8','9','T','J'], 
                        ['9','T','J','Q'],['T','J','Q','K'], ['J','Q','K','A'] ]

    kkDoubleSuitedList = []
    for i in range(len(face)):
        for j in range(len(face)):
            kkDoubleSuitedList.append((["K", "K", face[i], face[j]]))
            
            
    # Hole Cards to Flop With
    aqxxs = []
    for i in range(len(high_face)):
        for j in range(len(face)):
                akxxs.append((["A", "Q", high_face[i], face[j]]))
                    
    axxxs = []
    for i in range(len(face)):
        for j in range(len(face)):
            for k in range(len(face)):
                axxxs.append((["A",face[i],face[j],face[k]]))

    #BASELINE METHODS
    #Assume no checking
    #Random
    #Max Bet
    #Always Fold
    #Always Call

    #Three moves: Bet/Call, Fold, Raise


    # suited as long as there are at least 2 of any one suit 
    def checkSuited(self, suitList):
        spadeCount = suitList.count(chr(9824)) 
        heartCount = suitList.count(chr(9829))
        diamondCount = suitList.count(chr(9830))
        clubCount = suitList.count(chr(9827))
        
        if(spadeCount or heartCount or diamondCount or clubCount >= 2):
            return True
        else: 
            return False
        
    # specifically check that if the hole cards contain an Ace, it is suited
    def checkAceSuited(self, rankList,suitList):
        rank_set = set(rankList) 
        
        ace_pos_list = [ i for i in range(len(rankList)) if rankList[i] == 'A' ]
        #print(ace_pos_list)
        
        if 'A' in rank_set: 
            ace_suit_list = [ suitList[i] for i in range(len(rankList)) if rankList[i] == 'A' ]
            
            for elem in ace_suit_list:
                #print(elem)
                if (suitList.count(elem) >= 2):
                    return True
            
        else: 
            return True
        
        
    # double suited if no suit counts are zero or odd and sum of suit counts == 4
    def checkDoubleSuited(self, suitList):
        spadeCount = suitList.count(chr(9824))
        heartCount = suitList.count(chr(9829))
        diamondCount = suitList.count(chr(9830))
        clubCount = suitList.count(chr(9827))
        
        if(
        
            2 in {spadeCount, heartCount, diamondCount, clubCount} and ((spadeCount + heartCount == 4) or 
            (spadeCount + diamondCount == 4) or (spadeCount + clubCount == 4) or
            (heartCount + diamondCount == 4) or (heartCount + clubCount == 4) or (diamondCount + clubCount == 4))
            
        ):
            return True
        else:
            return False

    # check if at least 2 cards from the hole cards are consecutive
    def checkConnectedCards(self, rankList):
        for elem in rankList:
            idx = rankList.index(elem)
            face_idx = self.face.index(elem)
            
            #if(self.face[face_idx + 1] in rankList):
            if(self.face[face_idx] in rankList):
                return True
                
            else:
                return False



        
    # optimal strat
    def optimalStrat(self, obs):

        holeCards = obs["hole_cards"]

        rankList = []
        suitList = []

        for card in holeCards:
            rank_str = card.__str__()[0]
            rankList.append(rank_str)
        
            suit_str = card.__str__()[1]
            suitList.append(suit_str)
        
        #print("printing suitlist")
        #print(suitList)
        #print("rankList",rankList)
        # When to Raise
        if (self.checkDoubleSuited(suitList) == True):
            #print("we've passed the first if test")
            for elem in self.top30HandsList:
                if(set(rankList) == set(elem)):
                    return(obs["max_raise"])
            for elem in self.kkDoubleSuitedList:
                if(set(rankList) == set(elem)):
                    return(obs["max_raise"])
            for elem in self.highFourInARowList:
                if(set(rankList) == set(elem)):
                    return(obs["max_raise"])
        
        if(self.checkSuited(suitList) == True):
            #print("we're in the second if test")
            for elem in self.akxxs:
                #print(elem)
                if(set(rankList) == set(elem)):
                    #print("WEVE PASSED THE SET = SET TEST")
                    return(obs["max_raise"])
            #print("were done with the for loop in the second if test")
                
        # When to"Limp"
        if(self.checkAceSuited(rankList,suitList)):
            #print("were in the third if test")
            for elem in self.aqxxs:
                if(set(rankList) == set(elem)):
                    return(obs["call"])
                
            if(self.checkConnectedCards(rankList)):
                for elem in self.axxxs:
                    if(set(rankList) == set(elem)):
                        return(obs["call"])
        
        if(self.checkAceSuited(rankList,suitList)==False and self.checkDoubleSuited(suitList)==False and self.checkSuited(suitList)==False):
            #print("we're in the fourth if test")
            for elem in self.lowFourInARowList:
                if(set(rankList) == set(elem)):
                    return(obs["call"])
        
        else: 
            #print("we're folding according to optimal")
            return self.alwaysFold()


    def act(self, obs):
        if self.training:
            if obs["action"] == 0: #player 1
                #print("doing RL for player 0")
                return self.forwardRL(obs)
            else:
                #print("doing RL for player 1")
                #return self.randomAction(obs)
                return self.forwardRL(obs)
                #return self.alwaysBetCall(obs)
                #return self.alwaysMaxRaise(obs)
                #return self.optimalStrat(obs)
        else:
            #return self.bestTrainedAction(obs)
            
            if obs["action"] == 0:
                return self.bestTrainedAction(obs) #RL agent
                #return self.optimalStrat(obs)
                #return self.alwaysMaxRaise(obs)
            else:
                #return self.randomAction(obs)
                #return self.alwaysBetCall(obs)
                #return self.alwaysMaxRaise(obs)
                #return self.alwaysFold()
                #return self.bestTrainedAction(obs)
                return self.optimalStrat(obs)