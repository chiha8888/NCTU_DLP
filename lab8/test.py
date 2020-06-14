import gym

# class A:
#     def __init__(self):
#         self.vari='a'
#
# if __name__=='__main__':
#     a=A()
#     print(a.vari)
#
#     b=a
#     print(b.vari)
#     b.vari='b'
#     print(b.vari)
#
#     print(a.vari)

if __name__=='__main__':
    env = gym.make('LunarLanderContinuous-v2')
    state=env.reset()
    for _ in range(5):
        action=env.action_space.sample()
        print(action)
        state,reward,done,info=env.step(action)
        print(state)