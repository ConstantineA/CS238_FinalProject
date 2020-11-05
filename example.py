import gym

import clubs

import random
import numpy as np
import time

#GIT REPO WORKING

#env = gym.make("PotLimitOmahaTwoPlayer-v0")

#print("PAST MAKING THE ENVIRONMENT")

#env.register_agents([clubs.agent.baselineOmahaAgent.BaselineOmahaAgent()] *2)
#singleAgent = clubs.agent.testingRLOmahaAgent.TestingRLOmahaAgent()
#env.register_agents([singleAgent,singleAgent])
#env.register_agents([clubs.agent.testingRLOmahaAgent.TestingRLOmahaAgent()] *2)

#print("AFTER REGISTERING AGENTS")

#print(env.observation_space)

#print("observation space elements:")
#for i in env.observation_space:
 #   print(i)


#obs = env.reset()

#print("--------------------------------------")
#print("observations:")
#print(obs)
#print("--------------------------------------")
#print(obs["hole_cards"])
#print("player 0 hold cards",obs["hole_cards"][0])
#print("player 1 hold cards",obs["hole_cards"][1])
#print(get_rank_class(obs["hole_cards"][0]))


start_time = time.time()

#Xavier initialization
thetaBetCheck = np.random.randn(119)/np.sqrt(119)
thetaMaxRaise = np.random.randn(119)/np.sqrt(119)
thetaFold = np.random.randn(119)/np.sqrt(119)

#thetaBetCheck = 1/119 * np.ones(119)
#thetaMaxRaise = 1/119 * np.ones(119)
#thetaFold = 1/119 * np.ones(119)

thetasPlayer0 = np.array([thetaBetCheck, thetaMaxRaise, thetaFold])
thetasPlayer1 = np.array([thetaBetCheck, thetaMaxRaise, thetaFold])

#REMEMBER env.reset() just deals a new hand. you don't start from the beginning



#print("PAST MAKING THE ENVIRONMENT")

#env.register_agents([clubs.agent.baselineOmahaAgent.BaselineOmahaAgent()] *2)
singleAgent = clubs.agent.testingRLOmahaAgent.TestingRLOmahaAgent()

singleAgent.setStartingThetas(thetasPlayer0)

#TRAINING PHASE
#number of nights/tasks to continuously run
n = 5000
k = 100 #number of games in 1 night = 1 task
wins = 0
total = 0
for j in range(n):
    #print("--------------------------------------")
    if (j % 1000 == 0):
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

        while True:
            #print(obs)

            bet = env.act(obs)
            obs, rewards, done, info = env.step(bet)
            #print(obs)
            #print(rewards)

            if all(done):
                playerNumber = -1
                for pos in range(2):
                    if obs["active"][pos] == True:
                        playerNumber = pos
                #print(playerNumber)
                #print(rewards)
                env.agents[playerNumber].setRewardandUpdate(obs,rewards[playerNumber])
                break


            if not np.all(obs["stacks"]):
                noMoreStack = True
                break

            
            #print("rewards: ", rewards)
            playerNumber = obs["action"]
            inputReward = rewards[playerNumber]
            #print("input rewards for player: ", playerNumber, "is:", inputReward)
            #if playerNumber == 0:
            env.agents[playerNumber].setRewardandUpdate(obs,rewards[playerNumber])
            

            if playerNumber == 0:
                thetasPlayer0 = env.agents[playerNumber].getThetas()
            else:
                thetasPlayer1 = env.agents[playerNumber].getThetas()

        #at this point, if a player doesn't have any more chips, end the night (i.e. no more rounds for the night)
        if not np.all(obs["stacks"]):
                noMoreStack = True
        if noMoreStack:
            break
        #print(obs['stacks'])
        #print("END OF CURRENT GAME in the night ****************************************")
        #print(obs["stacks"])
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
print("Winning Night percentage (during training): ", wins/total)

print("EVALUATING AGAINST RANDOM")

trainedAgent = clubs.agent.testingRLOmahaAgent.TestingRLOmahaAgent()

trainedAgent.setStartingThetas(singleAgent.getThetas())
trainedAgent.setTrainingOff()

#randomAgent = clubs.agent.baselineOmahaAgent.BaselineOmahaAgent()



#number of nights/tasks to continuously run
n = 500
k = 100 #number of games in 1 night = 1 task
winsEvaluation = 0
totalEvaluation = 0
totalProfit = 0

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
                break

            if not np.all(obs["stacks"]):
                noMoreStack = True
                break
            #print(obs["stacks"])

        #at this point, if a player doesn't have any more chips, end the night (i.e. no more rounds for the night)
        if noMoreStack:
            break

        #print(obs["stacks"])
        obs = env.reset()
        
    #print(obs["stacks"])
    totalEvaluation += 1
    if (obs["stacks"][0] > obs["stacks"][1]):
        winsEvaluation +=1
    totalProfit += obs["stacks"][0] - 200
    #print("END OF CURRENT GAME")
    
print("Winning Night percentage (during evaluation against random): ", winsEvaluation/totalEvaluation)
print("Total profit: ", totalProfit)
print("Profit per Game Night: ", totalProfit/n)
print("Profit per individual game of Omaha: ", totalProfit/ (n*k))

end_time = time.time()
print("time it took:")
print(end_time - start_time)



