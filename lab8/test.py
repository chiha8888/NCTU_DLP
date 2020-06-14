import gym


# print(tuple(map(tuple,('a',',b','cv'))))
# env = gym.make('LunarLander-v2')
# for _ in range(20):
#     print(type(env.action_space.sample()))
class A:
    def __init__(self):
        self.vari='a'

if __name__=='__main__':
    a=A()
    print(a.vari)

    b=a
    print(b.vari)
    b.vari='b'
    print(b.vari)

    print(a.vari)