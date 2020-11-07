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

#REMEMBER env.reset() just deals a new hand. you don't start from the beginning



singleAgent = clubs.agent.testingRLOmahaAgent.TestingRLOmahaAgent()

singleAgent.setStartingThetas(thetasPlayer0)
#singleAgent.setStartingThetas(np.loadtxt("newSelfPlayThetas.txt"))

#TRAINING PHASE
#number of nights/tasks to continuously run
n = 1000
k = 100 #number of games in 1 night = 1 task
wins = 0
total = 0
for j in range(n):
    #print("--------------------------------------")
    if (j % 100 == 0):
        print("Game Night Number (Training): ", j)
    env = gym.make("PotLimitOmahaTwoPlayer-v0")
    env.register_agents([singleAgent,singleAgent])
    obs = env.reset() #first hand of the night is dealt

    
    #print(obs["stacks"])
    noMoreStack = False
    #print(thetasPlayer0[0])
    for i in range(k): # Each i represents 1 of the 100 rounds of poker played in 1 night
        #print("iteration number:",i)
        #print("in range of iterations")
        #print(obs)

        player0HoleCards =[]
        player1HoleCards = []

        while True:
            #print(obs)

            bet = env.act(obs)
            obs, rewards, done, info = env.step(bet)

            if not player0HoleCards:
                if obs["action"] == 0:
                    player0HoleCards = obs["hole_cards"]

            if not player1HoleCards:
                if obs["action"] == 1:
                    player1HoleCards = obs["hole_cards"]
            #print(obs)
            #print(rewards)

            if all(done):
                if not np.all(obs["stacks"]):
                    noMoreStack = True

                playerNumber = 1 if obs["hole_cards"] == player1HoleCards else 0
                #print(playerNumber)
                #print(rewards)
                #if (playerNumber == 0): #since we're training against optimal, don't do update function on non-RL agent
                env.agents[playerNumber].setRewardandUpdate(obs,rewards[playerNumber])
                break



            
            #print("rewards: ", rewards)
            playerNumber = obs["action"]
            inputReward = rewards[playerNumber]
            #print("input rewards for player: ", playerNumber, "is:", inputReward)
            #if playerNumber == 0: #since we're training against optimal, don't do update function on non-RL agent
            env.agents[playerNumber].setRewardandUpdate(obs,rewards[playerNumber])
            

            if playerNumber == 0:
                thetasPlayer0 = env.agents[playerNumber].getThetas()
            else:
                thetasPlayer1 = env.agents[playerNumber].getThetas()

        #at this point, if a player doesn't have any more chips, end the night (i.e. no more rounds for the night)
        if noMoreStack:
            break

        obs = env.reset()

    #print("Current stack",obs["stacks"])
    singleAgent.setStartingThetas(thetasPlayer0)
    #print(singleAgent.getThetas())
    #print("---------------------")

    total += 1
    if (obs["stacks"][0] > obs["stacks"][1]):
        wins +=1
    

#print(thetasPlayer0)
#print(thetasPlayer1)
print(np.array_equal(thetasPlayer0,thetasPlayer1))
print(np.array_equal(thetasPlayer0, singleAgent.getThetas()))
print("Winning Night percentage (during training): ", wins/total)

print("EVALUATING")



trainedAgent = clubs.agent.testingRLOmahaAgent.TestingRLOmahaAgent()

#trainedAgent.setStartingThetas(np.loadtxt("newSelfPlayThetas.txt"))
#trainedAgent.setStartingThetas(np.loadtxt("newTrainedOnOptimalThetas.txt"))
#trainedAgent.setStartingThetas(np.loadtxt("currentThetas.txt"))
#trainedAgent.setStartingThetas(np.loadtxt("thetasTrainedFromOptimal.txt"))
trainedAgent.setStartingThetas(singleAgent.getThetas())
trainedAgent.setTrainingOff()

#randomAgent = clubs.agent.baselineOmahaAgent.BaselineOmahaAgent()



#number of nights/tasks to continuously run
#n = 1000
#for main testing using self play thetas
n = 1000
k = 100 #number of games in 1 night = 1 task
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

    #k = 100 #number of games in 1 night = 1 task
    #print(obs["stacks"])
    #noMoreStack = False
    #print(thetasPlayer0[0])
    for i in range(k): # Each i represents 1 of the 100 rounds of poker played in 1 night
        #print("iteration number:",i)
        #print("in range of iterations")
        #print(obs)

        
    
        while True:

            bet = env.act(obs)
            obs, rewards, done, info = env.step(bet)
            #print(obs)
            

            if all(done):
                if not np.all(obs["stacks"]):
                    noMoreStack = True
                break


        #at this point, if a player doesn't have any more chips, end the night (i.e. no more rounds for the night)
        if noMoreStack:
            break


        obs = env.reset()
        


    #print("End of Night: ",obs["stacks"])
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



