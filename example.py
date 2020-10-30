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


k = 5

for i in range(k):
    #print(obs)
    while True:
        bet = env.act(obs)
        obs, rewards, done, info = env.step(bet)

        playerNumber = obs["action"]
        if (playerNumber >= 0):
            env.agents[playerNumber].sayHello(playerNumber)
            env.agents[playerNumber].setRewardandUpdate(obs,rewards[playerNumber])
        
        print(obs)

        #for (i, agent) in env.agents.items():
        #    if obs["action"] == i: #obs[action] is the player
         #       print(obs)
         #       agent.sayHello(i)
        print("-------------------------")
            #print(rewards[i])
            #agent.setRewardandUpdate(rewards[i]) #i give it
            

        #print(obs)
        #print("rewards: ", rewards)

        #print(env.agents[0])
        #env.agents[0].sayHello()

        if all(done):
            break

    print(i)
    print("Current stack",obs["stacks"])
    print("------")

    obs = env.reset()


