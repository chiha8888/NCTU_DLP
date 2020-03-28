import math
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """ Sigmoid function.
    This function accepts any shape of np.ndarray object as input and perform sigmoid operation.
    """
    return 1 / (1 + np.exp(-x))

def der_sigmoid(y):
    """ First derivative of Sigmoid function.
    The input to this function should be the value that output from sigmoid function.
    """
    return y * (1 - y)

class GenData:
    @staticmethod
    def _gen_linear(n=100):
        """ Data generation (Linear)

        Args:
            n (int):    the number of data points generated in total.

        Returns:
            data (np.ndarray, np.float):    the generated data with shape (n, 2). Each row represents
                a data point in 2d space.
            labels (np.ndarray, np.int):    the labels that correspond to the data with shape (n, 1).
                Each row represents a corresponding label (0 or 1).
        """
        data = np.random.uniform(0, 1, (n, 2))

        inputs = []
        labels = []

        for point in data:
            inputs.append([point[0], point[1]])

            if point[0] > point[1]:
                labels.append(0)
            else:
                labels.append(1)

        return np.array(inputs), np.array(labels).reshape((-1, 1))

    @staticmethod
    def _gen_xor(n=100):
        """ Data generation (XOR)

        Args:
            n (int):    the number of data points generated in total.

        Returns:
            data (np.ndarray, np.float):    the generated data with shape (n, 2). Each row represents
                a data point in 2d space.
            labels (np.ndarray, np.int):    the labels that correspond to the data with shape (n, 1).
                Each row represents a corresponding label (0 or 1).
        """
        data_x = np.linspace(0, 1, n // 2)

        inputs = []
        labels = []

        for x in data_x:
            inputs.append([x, x])
            labels.append(0)

            if x == 1 - x:
                continue

            inputs.append([x, 1 - x])
            labels.append(1)

        return np.array(inputs), np.array(labels).reshape((-1, 1))

    @staticmethod
    def fetch_data(mode, n):
        """ Data gather interface

        Args:
            mode (str): 'Linear' or 'XOR', indicate which generator is used.
            n (int):    the number of data points generated in total.
        """
        assert mode == 'Linear' or mode == 'XOR'

        data_gen_func = {
            'Linear': GenData._gen_linear,
            'XOR': GenData._gen_xor
        }[mode]

        return data_gen_func(n)


class SimpleNet:
    def __init__(self, hidden_size, lr=0.01, num_step=2000, print_interval=1):
        """ A hand-crafted implementation of simple network.

        Args:
            hidden_size: a tuple(size1,size2) of the number of hidden neurons used in each layer.
            lr: learning rate
            num_step (optional):    the total number of training steps.
            print_interval (optional):  the number of steps between each reported number.
        """
        self.EPS=1e-3
        self.hidden_size=hidden_size
        self.lr=lr
        self.num_step = num_step
        self.print_interval = print_interval
        
        # Model parameters initialization
        # Please initiate your network parameters here.
        '''
            X==a0 -> W1 -> b1 -> z1 -> a1 -> W2 -> b2 -> z2 -> a2 -> W3 -> b3 -> z3-> a3==y
        '''
        self.W=[None,np.random.randn(hidden_size[0],2),np.random.randn(hidden_size[1],hidden_size[0]),np.random.randn(1,hidden_size[1])]
        self.b=[None,np.zeros((hidden_size[0],1)),np.zeros((hidden_size[1],1)),np.zeros((1,1))]
        self.z=[None,np.zeros((hidden_size[0],1)),np.zeros((hidden_size[1],1)),np.zeros((1,1))]
        self.a=[None,np.zeros((hidden_size[0],1)),np.zeros((hidden_size[1],1)),np.zeros((1,1))]
    
    def init_W(self,n,m):
        """
        Returns:
            output(np.ndarray): (n*m) ndarray
        """
        init_epsilon=math.sqrt(6)/math.sqrt(n+m)
        W=np.random.rand(n,m)*(2*init_epsilon)-init_epsilon
        return W
    
    def compute_loss(self,gt_y,pred_y):
        """
        logistic regression, use cross entropy as loss value 
        """
        loss=-(1/gt_y.shape[1])*(gt_y@np.log(pred_y+self.EPS).T+(1-gt_y)@np.log(1-pred_y+self.EPS).T)
        return float(loss)
    
    def forward(self, inputs):
        """ Implementation of the forward pass.
        It should accepts the inputs and passing them through the network and return results.
        Args:
            inputs: (2*batch_size) ndarray
        Returns:
            output: (1*batch_size) ndarray
        """
        self.a[0]=inputs
        # hidden layer 1
        self.z[1]=self.W[1]@self.a[0]+self.b[1]
        self.a[1]=sigmoid(self.z[1])
        # hidden layer 2
        self.z[2]=self.W[2]@self.a[1]+self.b[2]
        self.a[2]=sigmoid(self.z[2])
        # output layer 
        self.z[3]=self.W[3]@self.a[2]+self.b[3]
        self.a[3]=sigmoid(self.z[3])
        
        return self.a[3]

    def backward(self,gt_y,pred_y):
        """ Implementation of the backward pass.
        It should utilize the saved loss to compute gradients and update the network all the way to the front.
        Args:
            gt_y: (1*batch_size) ndarray
            pred_y: (1*batch_size) ndarray
        """
        batch_size=gt_y.shape[1]
        # bp
        grad_a3=-(gt_y/(pred_y+self.EPS)-(1-gt_y)/(1-pred_y+self.EPS))
        grad_z3=grad_a3*der_sigmoid(self.a[3])
        grad_W3=grad_z3@self.a[2].T*(1/batch_size)
        grad_b3=np.sum(grad_z3,axis=1,keepdims=True)*(1/batch_size)
        
        grad_a2=self.W[3].T@grad_z3
        grad_z2=grad_a2*der_sigmoid(self.a[2])
        grad_W2=grad_z2@self.a[1].T*(1/batch_size)
        grad_b2=np.sum(grad_z2,axis=1,keepdims=True)*(1/batch_size)
            
        grad_a1=self.W[2].T@grad_z2
        grad_z1=grad_a1*der_sigmoid(self.a[1])
        grad_W1=grad_z1@self.a[0].T*(1/batch_size)
        grad_b1=np.sum(grad_z1,axis=1,keepdims=True)*(1/batch_size)
        
        # update
        self.W[1]-=self.lr*grad_W1
        self.W[2]-=self.lr*grad_W2
        self.W[3]-=self.lr*grad_W3
        self.b[1]-=self.lr*grad_b1
        self.b[2]-=self.lr*grad_b2
        self.b[3]-=self.lr*grad_b3
        
        return

    def train(self, X, y):
        """ The training routine that runs and update the model.
        Args:
            X: (2,batch_size) ndarray
            y: (1,batch_size) ndarray
        """
        # make sure that the amount of data and label is match
        assert X.shape[1] == y.shape[1]

        for epochs in range(self.num_step):
            #   1. forward passing
            #   2. compute loss
            #   3. propagate gradient backward to the front
            pred_y=self.forward(X)
            self.backward(y,pred_y)
            
            if epochs % self.print_interval == 0:
                loss=self.compute_loss(y,pred_y)
                acc=(1.-np.sum(np.abs(y-np.round(pred_y)))/y.shape[1])*100
                print(f'Epochs {epochs}: loss={loss:.5f} accuracy={acc:.2f}%')

    def test(self,X,y):
        """ The testing routine that run forward pass and report the accuracy.
        Args:
            X: (2,batch_size) ndarray
            y: (1,batch_size) ndarray
        """
        pred_y=self.forward(X)
        loss=self.compute_loss(y,pred_y)
        acc=(1.-np.sum(np.abs(y-np.round(pred_y)))/y.shape[1])*100
        print(f'loss={loss:.5f} accuracy={acc:.2f}%')
    
    @staticmethod
    def plot_result(data, gt_y, pred_y):
        """ Data visualization with ground truth and predicted data comparison. There are two plots
        for them and each of them use different colors to differentiate the data with different labels.

        Args:
            data:   the input data
            gt_y:   ground truth to the data
            pred_y: predicted results to the data
        """
        assert data.shape[0] == gt_y.shape[0]
        assert data.shape[0] == pred_y.shape[0]

        plt.figure()

        plt.subplot(1, 2, 1)
        plt.title('Ground Truth', fontsize=18)

        for idx in range(data.shape[0]):
            if gt_y[idx] == 0:
                plt.plot(data[idx][0], data[idx][1], 'ro')
            else:
                plt.plot(data[idx][0], data[idx][1], 'bo')

        plt.subplot(1, 2, 2)
        plt.title('Prediction', fontsize=18)

        for idx in range(data.shape[0]):
            if pred_y[idx] == 0:
                plt.plot(data[idx][0], data[idx][1], 'ro')
            else:
                plt.plot(data[idx][0], data[idx][1], 'bo')

        plt.show()

if __name__=='__main__':
    X,y= GenData.fetch_data('Linear', 70)
    X=X.T
    y=y.T

    net = SimpleNet((10,10), lr=0.1,num_step=30000,print_interval=1)
    print('start training:')
    net.train(X,y)
    print('training finished\n')
    print('start testing:')
    net.test(X,y)
    print('testing finished')

    pred_result=net.forward(X)
    SimpleNet.plot_result(X.T,y.T,np.round(pred_result).T)