import gym

print(tuple(map(tuple,('a',',b','cv'))))
env = gym.make('LunarLander-v2')
for _ in range(20):
    print(type(env.action_space.sample()))
