import gym

import clubs

import random

#from . import base
from collections import defaultdict
import math
import numpy as np
from typing import Dict, List, Optional, Union

from clubs import error

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
        face_idx = face.index(elem)
        
        if(face[face_idx + 1] in rankList):
            return True
            
        else:
            return False

# fold observation
def alwaysFold(self):
    return -1

    
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
    
    # When to Raise
    if (checkDoubleSuited(suitList) == True):
        for elem in top30HandsList:
            if(set(rankList) == set(elem)):
                return(obs["max_raise"])
        for elem in kkDoubleSuitedList:
            if(set(rankList) == set(elem)):
                return(obs["max_raise"])
        for elem in highFourInARowList:
            if(set(rankList) == set(elem)):
                return(obs["max_raise"])
    
    elif(checkSuited(suitList) == True):
        for elem in akxxs:
            if(set(rankList) == set(elem)):
                return(obs["max_raise"])
            
    # When to"Limp"
    elif(checkAceSuited(rankList,suitList)):
        for elem in aqxxs:
            if(set(rankList) == set(elem)):
                return(obs["call"])
            
        if(checkConnectedCards(rankList)):
            for elem in axxxs:
                if(set(rankList) == set(elem)):
                    return(obs["call"])
    
    elif(checkAceSuited(rankList,suitList)==False and checkDoubleSuited(suitList)==False and checkSuited(suitList)==False):
        for elem in lowFourInARowList:
            if(set(rankList) == set(elem)):
                return(obs["call"])
    
    else: 
        alwaysFold(self)