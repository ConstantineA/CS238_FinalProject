import gym

import clubs

import random

#GIT REPO WORKING

env = gym.make("PotLimitOmahaTwoPlayer-v0")

print("PAST MAKING THE ENVIRONMENT")

env.register_agents([clubs.agent.baselineOmahaAgent.BaselineOmahaAgent()] *2)

print("AFTER REGISTERING AGENTS")

print(env.observation_space)

for i in env.observation_space:
    print(i)


obs = env.reset()


for i in range(10):
    while True:
        bet = env.act(obs)
        obs, rewards, done, info = env.step(bet)

        if all(done):
            break

    print(i)
    print("Current stack",obs["stacks"])
    print("------")

    obs = env.reset()


