import matplotlib.pyplot as plt

def plot(loss,bleu):
    fig=plt.figure(figsize=(8,6))
    plt.plot(loss,label='loss')
    plt.plot(bleu,label='BLEU-4')
    plt.legend()
    return fig