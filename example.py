import gym

import clubs

import random

#GIT REPO WORKING

env = gym.make("PotLimitOmahaTwoPlayer-v0")

#print("PAST MAKING THE ENVIRONMENT")

#env.register_agents([clubs.agent.baselineOmahaAgent.BaselineOmahaAgent()] *2)
env.register_agents([clubs.agent.testingRLOmahaAgent.TestingRLOmahaAgent()] *2)

#print("AFTER REGISTERING AGENTS")

#print(env.observation_space)

print("observation space elements:")
for i in env.observation_space:
    print(i)


obs = env.reset()

print("--------------------------------------")
print("observations:")
print(obs)
print("--------------------------------------")
print(obs["hole_cards"])
print(obs["hole_cards"][0])
#print(get_rank_class(obs["hole_cards"][0]))


k = 10

for i in range(k):
    print(i)
    #print("in range of iterations")
    #print(obs)
    while True:
        #print("inside while loop")
        bet = env.act(obs)
        #print("just finished bet")
        obs, rewards, done, info = env.step(bet)
        #print("just finished step")

        playerNumber = obs["action"]
        #print("playernumber:", playerNumber)
        #if (playerNumber >= 0):
         #   print("within rewardUpdate step")
            #env.agents[playerNumber].sayHello(playerNumber)
        env.agents[playerNumber].setRewardandUpdate(obs,rewards[playerNumber])
        
        #print(obs)

        #for (i, agent) in env.agents.items():
        #    if obs["action"] == i: #obs[action] is the player
         #       print(obs)
         #       agent.sayHello(i)
        #print("-------------------------")
            #print(rewards[i])
            #agent.setRewardandUpdate(rewards[i]) #i give it
            

        #print(obs)
        #print("rewards: ", rewards)

        #print(env.agents[0])
        #env.agents[0].sayHello()

        if all(done):
            break

    #print(i)
    #print("Current stack",obs["stacks"])
    #print("------")
    if (i == k-1): #the last iteration
        print("Current stack",obs["stacks"])
    obs = env.reset()

print("Current stack",obs["stacks"])


