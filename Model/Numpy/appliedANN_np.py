import numpy as np
import scipy as sp
#import wandb
import time, sys

sys.path.append(
    "/home/ratnamaru/Documents/Acads/OnlineCourses/GITHUB/MY_REPOS/ANNS_ODES_PDES/"
)
from Utils.Numpy.activation import *


class AppliedANN:
    def __init__(
        self, 
        polynomialCoeffs,       
        optimizer,
        weight_decay,
        learning_rate,
        max_epochs,
        activation,
        initializer,
        loss

    ):

        self.polynomialCoeffs = polynomialCoeffs
        self.polynomialDegree = len(polynomialCoeffs) - 1
        # self.layers = layers
        self.layers = [1, 1, self.polynomialDegree, 1]


        self.Activations_dict = {"SIGMOID": sigmoid, "TANH": tanh, "RELU": relu, "POLY":self.polynomialBasis}
        self.DerActivation_dict = {
            "SIGMOID": der_sigmoid,
            "TANH": der_tanh,
            "RELU": der_relu,
            "POLY":self.der_polynomialBasis
        }

        self.Initializer_dict = {
            "XAVIER": self.Xavier_initializer,
            "RANDOM": self.random_initializer,
            "HE": self.He_initializer,
            "ROOTFINDER": self.polynomialRootFinder_initializer
        }

        self.Optimizer_dict = {
            "SGD": self.sgd
        }
        
        self.activation = self.Activations_dict[activation]
        self.der_activation = self.DerActivation_dict[activation]
        self.optimizer = self.Optimizer_dict[optimizer]
        if initializer == "ROOTFINDER":
             self.weights, self.biases = self.polynomialRootFinder_initializer()
        
        else:
            self.initializer = self.Initializer_dict[initializer]
            self.weights, self.biases = self.initializeNeuralNet(self.layers)
        self.loss_function = loss
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        
        


        
        
    # helper functions
    def oneHotEncode(self, Y_train_raw):
        Ydata = np.zeros((self.num_classes, Y_train_raw.shape[0]))
        for i in range(Y_train_raw.shape[0]):
            value = Y_train_raw[i]
            Ydata[int(value)][i] = 1.0
        return Ydata

    # Loss functions
    def meanSquaredErrorLoss(self, Y_true, Y_pred):
        MSE = np.mean((Y_true - Y_pred) ** 2)
        return MSE

    def crossEntropyLoss(self, Y_true, Y_pred):
        CE = [-Y_true[i] * np.log(Y_pred[i]) for i in range(len(Y_pred))]
        crossEntropy = np.mean(CE)
        return crossEntropy

    def L2RegularisationLoss(self, weight_decay):
        ALPHA = weight_decay
        return ALPHA * np.sum(
            [
                np.linalg.norm(self.weights[str(i + 1)]) ** 2
                for i in range(len(self.weights))
            ]
        )


    def accuracy(self, Y_true, Y_pred, data_size):
        Y_true_label = []
        Y_pred_label = []
        ctr = 0
        for i in range(data_size):
            Y_true_label.append(np.argmax(Y_true[:, i]))
            Y_pred_label.append(np.argmax(Y_pred[:, i]))
            if Y_true_label[i] == Y_pred_label[i]:
                ctr += 1
        accuracy = ctr / data_size
        return accuracy, Y_true_label, Y_pred_label

    def Xavier_initializer(self, size):
        in_dim = size[1]
        out_dim = size[0]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return np.random.normal(0, xavier_stddev, size=(out_dim, in_dim))

    def random_initializer(self, size):
        in_dim = size[1]
        out_dim = size[0]
        return np.random.normal(0, 1, size=(out_dim, in_dim))


    def He_initializer(self,size):
        in_dim = size[1]
        out_dim = size[0]
        He_stddev = np.sqrt(2 / (in_dim))
        return np.random.normal(0, 1, size=(out_dim, in_dim)) * He_stddev


    def polynomialBasis(self, Z):
        f = [Z[i]**(i+1) for i in range(len(Z))]
        return np.array(f).reshape(Z.shape)

    def der_polynomialBasis(self, Z):
        df = [(i+1)*Z[i]**i for i in range(len(Z))]
        return np.array(df).reshape(Z.shape)
        
        
        
    def polynomialRootFinder_initializer(self):
        weights = {}
        biases = {}
        
        weights['1'] = np.random.normal(0, 1, size=(self.layers[0], self.layers[1])) 
        biases['1'] = np.zeros((self.layers[1],1))
        weights['2'] = np.ones((self.layers[1], self.layers[2]))
        biases['2'] = np.zeros((self.layers[2],1))
        weights['3'] = np.array(self.polynomialCoeffs[1:]).reshape(self.layers[2], self.layers[3]) 
        biases['3'] = np.array(self.polynomialCoeffs[0]).reshape(self.layers[3],1)
        
        return weights, biases
        

    def initializeNeuralNet(self, layers):
        weights = {}
        biases = {}
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.initializer(size=[layers[l + 1], layers[l]])
            b = np.zeros((layers[l + 1], 1))
            weights[str(l + 1)] = W
            biases[str(l + 1)] = b
        return weights, biases

    
    def forwardPolynomialRootFinderANN(self):
        # Number of layers = length of weight matrix + 1

        H = {}
        A = {}
        A["0"] = np.array(1).reshape(1,1)
        H["0"] = A["0"]
        A["1"] = np.add(np.matmul(self.weights["1"].transpose(), H["0"]), self.biases["1"])
        H["1"] = A["1"]
        A["2"] = np.add(np.matmul(self.weights["2"].transpose(), H["1"]), self.biases["2"])
        H["2"] = self.Activations_dict["POLY"](A["2"]) 
        A["3"] = np.add(np.matmul(self.weights["3"].transpose(), H["2"]), self.biases["3"])
        H["3"] = self.Activations_dict["TANH"](A["3"]) 
        Y = H["3"]
        return Y, H, A
    

    def backPropagatePolynomialANN(
        self, Y, H, A):

        gradients_weights = []
        num_layers = len(self.layers)

        globals()["grad_W1"] = ( -Y*(1- Y**2)*np.sum([self.polynomialCoeffs[i]*H["2"][i-1] if i > 0 else self.polynomialCoeffs[0] for i in range(len(H["2"]))]))
        gradients_weights.append(globals()["grad_W1"])

        return gradients_weights

    #Optimisers defined here onwards
    def sgd(self, epochs, learning_rate):
        
        trainingloss = []
        

        for epoch in range(epochs):
            start_time = time.time()
            
            LOSS = []


            Y, H, A = self.forwardPolynomialRootFinderANN()
            grad_weights = self.backPropagatePolynomialANN(Y, H, A)
            deltaw = grad_weights

            LOSS.append(self.meanSquaredErrorLoss( 0, Y))
            
            self.weights["1"] = self.weights["1"] - learning_rate * deltaw[0]

            elapsed = time.time() - start_time
            
            
            
            trainingloss.append(np.mean(LOSS))
            #trainingaccuracy.append(self.accuracy(Y_train, Y_pred, length_dataset)[0])
            #validationaccuracy.append(self.accuracy(self.Y_val, self.predict(self.X_val, self.N_val), self.N_val)[0])
            
            print(
                        "Epoch: %d, Loss: %.3e, Time: %.2f, Learning Rate: %.3e"
                        % (
                            epoch,
                            trainingloss[epoch],
                            elapsed,
                            self.learning_rate,
                        )
                    )

        
        Y_pred= self.forwardPolynomialRootFinderANN()[0]
            #wandb.log({'loss':np.mean(LOSS), 'trainingaccuracy':trainingaccuracy[epoch], 'validationaccuracy':validationaccuracy[epoch],'epoch':epoch, })
        # data = [[epoch, loss[epoch]] for epoch in range(epochs)]
        # table = wandb.Table(data=data, columns = ["Epoch", "Loss"])
        # wandb.log({'loss':wandb.plot.line(table, "Epoch", "Loss", title="Loss vs Epoch Line Plot")})
        return trainingloss, Y_pred #, trainingaccuracy, validationaccuracy, Y_pred



 

