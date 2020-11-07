import gym

import clubs

import random
import numpy as np
import time




start_time = time.time()

#Xavier initialization
thetaBetCheck = np.random.randn(119)/np.sqrt(119)
thetaMaxRaise = np.random.randn(119)/np.sqrt(119)
thetaFold = np.random.randn(119)/np.sqrt(119)



thetasPlayer0 = np.array([thetaBetCheck, thetaMaxRaise, thetaFold])
thetasPlayer1 = np.array([thetaBetCheck, thetaMaxRaise, thetaFold])

singleAgent = clubs.agent.testingRLOmahaAgent.TestingRLOmahaAgent()

singleAgent.setStartingThetas(thetasPlayer0)


#TRAINING PHASE
n = 1000 #number of nights/tasks to continuously run
k = 100 #number of games in 1 night = 1 task
wins = 0
total = 0
for j in range(n):
    if (j % 100 == 0):
        print("Game Night Number (Training): ", j)
    env = gym.make("PotLimitOmahaTwoPlayer-v0")
    env.register_agents([singleAgent,singleAgent])
    obs = env.reset() #first hand of the night is dealt

    noMoreStack = False

    for i in range(k): # Each i represents 1 of the 100 rounds of poker played in 1 night

        player0HoleCards =[]
        player1HoleCards = []

        while True:
            bet = env.act(obs)
            obs, rewards, done, info = env.step(bet)

            if not player0HoleCards:
                if obs["action"] == 0:
                    player0HoleCards = obs["hole_cards"]

            if not player1HoleCards:
                if obs["action"] == 1:
                    player1HoleCards = obs["hole_cards"]

            if all(done):
                if not np.all(obs["stacks"]):
                    noMoreStack = True

                playerNumber = 1 if obs["hole_cards"] == player1HoleCards else 0
                #print(playerNumber)
                #print(rewards)
                if (playerNumber == 0): #comment out when we're training against optimal, don't do update function on non-RL agent
                    env.agents[playerNumber].setRewardandUpdate(obs,rewards[playerNumber])
                break

            playerNumber = obs["action"]
            inputReward = rewards[playerNumber]

            if playerNumber == 0: #comment out when we're training against optimal, don't do update function on non-RL agent
                env.agents[playerNumber].setRewardandUpdate(obs,rewards[playerNumber])
            

            if playerNumber == 0:
                thetasPlayer0 = env.agents[playerNumber].getThetas()
            else:
                thetasPlayer1 = env.agents[playerNumber].getThetas()

        #at this point, if a player doesn't have any more chips, end the night (i.e. no more rounds for the night)
        if noMoreStack:
            break

        obs = env.reset()

    singleAgent.setStartingThetas(thetasPlayer0)

    total += 1
    if (obs["stacks"][0] > obs["stacks"][1]):
        wins +=1
    

print(np.array_equal(thetasPlayer0,thetasPlayer1))
print(np.array_equal(thetasPlayer0, singleAgent.getThetas()))
print("Winning Night percentage (during training): ", wins/total)


#EVALUATION PHASE
print("EVALUATING")

trainedAgent = clubs.agent.testingRLOmahaAgent.TestingRLOmahaAgent()

#trainedAgent.setStartingThetas(np.loadtxt("currentThetas.txt"))
#trainedAgent.setStartingThetas(np.loadtxt("thetasTrainedFromOptimal.txt"))
trainedAgent.setStartingThetas(singleAgent.getThetas())
trainedAgent.setTrainingOff()



n = 10000
k = 100
winsEvaluation = 0
totalEvaluation = 0
totalProfit = 0
opponentTotalProfit = 0
biggestWin = 0
biggestLoss = 0
noMoreStack = False
for j in range(n):
    if (j % 100 == 0):
        print("Game Night Number (Evaluation): --------------------------------------------", j)
    env = gym.make("PotLimitOmahaTwoPlayer-v0")
    env.register_agents([trainedAgent,trainedAgent])
    obs = env.reset() #first hand of the night is dealt

    for i in range(k): # Each i represents 1 of the 100 rounds of poker played in 1 night
    
        while True:

            bet = env.act(obs)
            obs, rewards, done, info = env.step(bet)

            if all(done):
                if not np.all(obs["stacks"]):
                    noMoreStack = True
                break

        #at this point, if a player doesn't have any more chips, end the night (i.e. no more rounds for the night)
        if noMoreStack:
            break

        obs = env.reset()
        

    totalEvaluation += 1
    if (obs["stacks"][0] > obs["stacks"][1]):
        winsEvaluation +=1
    thisGamesProfit = obs["stacks"][0] - 200
    thisOpponentTotalProfit = obs["stacks"][1] - 200
    if thisGamesProfit > biggestWin:
        biggestWin = thisGamesProfit
    if thisGamesProfit < biggestLoss:
        biggestLoss = thisGamesProfit
    totalProfit += thisGamesProfit
    opponentTotalProfit += thisOpponentTotalProfit
    

    #print("END OF CURRENT GAME")
    
print("Winning Night percentage (during evaluation against random): ", winsEvaluation/totalEvaluation)
print("Total profit: ", totalProfit)
print("Profit per Game Night: ", totalProfit/n)
print("Profit per individual game of Omaha: ", totalProfit/ (n*k))
print("Biggest Win:", biggestWin)
print("Biggest Loss:", biggestLoss)
print("Opponent's Total Profit:", opponentTotalProfit)

end_time = time.time()
print("time it took:")
print(end_time - start_time)

np.savetxt("thetasWithPotFeature.txt", trainedAgent.getThetas(), fmt="%s")



