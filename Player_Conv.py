import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from Board import * 
import numpy as np 
import gc 
from copy import deepcopy 
import tensorflow as tf 
#import input_sampling

reset = False
new_round = True 

def Rect(x):
    return tf.cast(np.max([0.01*x, x], axis = 0), tf.float32)

def dRect(x): 
    return tf.cast(x>0,tf.float32)+tf.cast(x<=0,tf.float32)*0.01 

def Sigmoid(x): #only accept size N,1 arrays 
    return tf.cast(1/(1+np.exp(np.min([np.ones(x.shape)*1E2,-x],axis = 0))), tf.float32)

def dSigmoid(x):
    return tf.cast(Sigmoid(x)*(1-Sigmoid(x)), tf.float32)

class Conv(): 
    def __init__(self):
        self.load() 
        
    def load(self, w_file = "w_RL"):
        dat = np.load(w_file+".npz", allow_pickle = True)
        self.w = dat['w'] 
        self.b = dat['b'] 
        
    def save_w(self, w_file = "w_RL"):
        np.savez(w_file+".npz", w = self.w, b = self.b)
    
    def forward(self, batch_input):
        N = batch_input.shape[0]
        self.x = batch_input
        self.z1 = tf.nn.conv2d(self.x, self.w[0], [1,1,1,1], "VALID") + tf.tile(tf.expand_dims(self.b[0], 0), [N,1,1,1])
        self.a1 = Rect(self.z1)
        self.z2 = tf.nn.conv2d(self.a1, self.w[1], [1,1,1,1], "SAME") + tf.tile(tf.expand_dims(self.b[1], 0), [N,1,1,1])
        self.a2 = Rect(self.z2)
        self.z3 = tf.nn.conv2d(self.a2, self.w[2], [1,1,1,1], "SAME") + tf.tile(tf.expand_dims(self.b[2], 0), [N,1,1,1])
        self.a3 = Rect(self.z3)
        self.z4 = tf.nn.conv2d(self.a3, self.w[3], [1,1,1,1], "SAME") + tf.tile(tf.expand_dims(self.b[3], 0), [N,1,1,1])
        self.a4 = Rect(self.z4)
        self.z5 = tf.nn.conv2d(self.a4, self.w[4], [1,1,1,1], "SAME") + tf.tile(tf.expand_dims(self.b[4], 0), [N,1,1,1])
        self.a5 = Rect(self.z5)
        self.z6 = tf.nn.conv2d(self.a5, self.w[5], [1,1,1,1], "SAME") + tf.tile(tf.expand_dims(self.b[5], 0), [N,1,1,1])
        self.a6 = Rect(self.z6)
        self.z7 = tf.nn.conv2d(self.a6, self.w[6], [1,1,1,1], "SAME") + tf.tile(tf.expand_dims(self.b[6], 0), [N,1,1,1])
        self.a7 = Rect(self.z7)
        self.z8 = tf.nn.conv2d(self.a7, self.w[7], [1,1,1,1], "SAME") + tf.tile(tf.expand_dims(self.b[7], 0), [N,1,1,1])
        self.a8 = Rect(self.z8)
        self.z9 = tf.nn.conv2d(self.a8, self.w[8], [1,1,1,1], "SAME") + tf.tile(tf.expand_dims(self.b[8], 0), [N,1,1,1])
        self.a9 = Rect(self.z9)
        self.z10 = tf.nn.conv2d(self.a9, self.w[9], [1,1,1,1], "SAME") + tf.tile(tf.expand_dims(self.b[9], 0), [N,1,1,1])
        self.a10 = Rect(self.z10)
        self.z11 = tf.nn.conv2d(self.a10, self.w[10], [1,1,1,1], "SAME") + tf.tile(tf.expand_dims(self.b[10], 0), [N,1,1,1])
        self.a11 = Rect(self.z11) 
        self.z12 = tf.nn.conv2d(self.a11, self.w[11], [1,1,1,1], "SAME") + tf.tile(tf.expand_dims(self.b[11], 0), [N,1,1,1])
        self.a12 = Rect(self.z12) 
        self.z13 = tf.nn.conv2d(self.a12, self.w[12], [1,1,1,1], "SAME") + tf.tile(tf.expand_dims(self.b[12], 0), [N,1,1,1])
        self.p = tf.nn.softmax(tf.reshape(self.z13, [-1, self.z13.shape[1]*self.z13.shape[2]]), axis = 1)
        self.p1 = tf.expand_dims(self.p, 2) 

    def loss_calc(self, inX, prob):
        self.forward(inX)
        self.loss = tf.reduce_sum(tf.multiply(-prob, np.log(np.exp(-50)+self.p)))

    def backward(self, inX, prob, eta = 0.01):
        self.loss_calc(inX, prob)
        N = prob.shape[0]
        DS = tf.compat.v1.matrix_diag(self.p) - tf.matmul(self.p1, tf.transpose(self.p1, [0,2,1]))
        #DS := dSigmoid(z4) 
        p1 = tf.expand_dims(tf.divide(-prob, self.p+np.exp(-50)), 2) 
        #p1 = (a-y)

        self.d13 = tf.reshape(tf.matmul(DS, p1), [-1, 19, 19, 1]) 

        self.d12 = tf.multiply(tf.nn.conv2d(self.d13,
                               tf.transpose(tf.image.rot90(
                                   tf.transpose(self.w[12], [2,0,1,3]), k=2),
                                   [1, 2, 3, 0]),
                               [1,1,1,1], "SAME"), dRect(self.z12))
        self.d11 = tf.multiply(tf.nn.conv2d(self.d12,
                               tf.transpose(tf.image.rot90(
                                   tf.transpose(self.w[11], [2,0,1,3]), k=2),
                                   [1, 2, 3, 0]),
                               [1,1,1,1], "SAME"), dRect(self.z11)) 
        self.d10 = tf.multiply(tf.nn.conv2d(self.d11,
                               tf.transpose(tf.image.rot90(
                                   tf.transpose(self.w[10], [2,0,1,3]), k=2),
                                   [1, 2, 3, 0]),
                               [1,1,1,1], "SAME"), dRect(self.z10))
        self.d9 = tf.multiply(tf.nn.conv2d(self.d10,
                               tf.transpose(tf.image.rot90(
                                   tf.transpose(self.w[9], [2,0,1,3]), k=2),
                                   [1, 2, 3, 0]),
                               [1,1,1,1], "SAME"), dRect(self.z9))
        self.d8 = tf.multiply(tf.nn.conv2d(self.d9,
                               tf.transpose(tf.image.rot90(
                                   tf.transpose(self.w[8], [2,0,1,3]), k=2),
                                   [1, 2, 3, 0]),
                               [1,1,1,1], "SAME"), dRect(self.z8))
        self.d7 = tf.multiply(tf.nn.conv2d(self.d8,
                               tf.transpose(tf.image.rot90(
                                   tf.transpose(self.w[7], [2,0,1,3]), k=2),
                                   [1, 2, 3, 0]),
                               [1,1,1,1], "SAME"), dRect(self.z7))
        self.d6 = tf.multiply(tf.nn.conv2d(self.d7,
                               tf.transpose(tf.image.rot90(
                                   tf.transpose(self.w[6], [2,0,1,3]), k=2),
                                   [1, 2, 3, 0]),
                               [1,1,1,1], "SAME"), dRect(self.z6))
        self.d5 = tf.multiply(tf.nn.conv2d(self.d6,
                               tf.transpose(tf.image.rot90(
                                   tf.transpose(self.w[5], [2,0,1,3]), k=2),
                                   [1, 2, 3, 0]),
                               [1,1,1,1], "SAME"), dRect(self.z5))
        self.d4 = tf.multiply(tf.nn.conv2d(self.d5,
                               tf.transpose(tf.image.rot90(
                                   tf.transpose(self.w[4], [2,0,1,3]), k=2),
                                   [1, 2, 3, 0]),
                               [1,1,1,1], "SAME"), dRect(self.z4))
        self.d3 = tf.multiply(tf.nn.conv2d(self.d4,
                               tf.transpose(tf.image.rot90(
                                   tf.transpose(self.w[3], [2,0,1,3]), k=2),
                                   [1, 2, 3, 0]),
                               [1,1,1,1], "SAME"), dRect(self.z3))
        self.d2 = tf.multiply(tf.nn.conv2d(self.d3,
                               tf.transpose(tf.image.rot90(
                                   tf.transpose(self.w[2], [2,0,1,3]), k=2),
                                   [1, 2, 3, 0]),
                               [1,1,1,1], "SAME"), dRect(self.z2))
        self.d1 = tf.multiply(tf.nn.conv2d(self.d2,
                               tf.transpose(tf.image.rot90(
                                   tf.transpose(self.w[1], [2,0,1,3]), k=2), 
                                   [1, 2, 3, 0]),
                               [1,1,1,1], "SAME"), dRect(self.z1))

        self.dW13 = tf.compat.v1.nn.conv2d_backprop_filter(self.a12, self.w[12].shape, self.d13, [1,1,1,1], "VALID")
        self.dW12 = tf.compat.v1.nn.conv2d_backprop_filter(tf.pad(self.a11,[[0,0],[1,1],[1,1],[0,0]]), self.w[11].shape, self.d12, [1,1,1,1], "VALID")
        self.dW11 = tf.compat.v1.nn.conv2d_backprop_filter(tf.pad(self.a10,[[0,0],[1,1],[1,1],[0,0]]), self.w[10].shape, self.d11, [1,1,1,1], "VALID")
        self.dW10 = tf.compat.v1.nn.conv2d_backprop_filter(tf.pad(self.a9,[[0,0],[1,1],[1,1],[0,0]]), self.w[9].shape, self.d10, [1,1,1,1], "VALID")
        self.dW9 = tf.compat.v1.nn.conv2d_backprop_filter(tf.pad(self.a8,[[0,0],[1,1],[1,1],[0,0]]), self.w[8].shape, self.d9, [1,1,1,1], "VALID")
        self.dW8 = tf.compat.v1.nn.conv2d_backprop_filter(tf.pad(self.a7,[[0,0],[1,1],[1,1],[0,0]]), self.w[7].shape, self.d8, [1,1,1,1], "VALID")
        self.dW7 = tf.compat.v1.nn.conv2d_backprop_filter(tf.pad(self.a6,[[0,0],[1,1],[1,1],[0,0]]), self.w[6].shape, self.d7, [1,1,1,1], "VALID")
        self.dW6 = tf.compat.v1.nn.conv2d_backprop_filter(tf.pad(self.a5,[[0,0],[1,1],[1,1],[0,0]]), self.w[5].shape, self.d6, [1,1,1,1], "VALID")
        self.dW5 = tf.compat.v1.nn.conv2d_backprop_filter(tf.pad(self.a4,[[0,0],[1,1],[1,1],[0,0]]), self.w[4].shape, self.d5, [1,1,1,1], "VALID")
        self.dW4 = tf.compat.v1.nn.conv2d_backprop_filter(tf.pad(self.a3,[[0,0],[1,1],[1,1],[0,0]]), self.w[3].shape, self.d4, [1,1,1,1], "VALID")
        self.dW3 = tf.compat.v1.nn.conv2d_backprop_filter(tf.pad(self.a2,[[0,0],[1,1],[1,1],[0,0]]), self.w[2].shape, self.d3, [1,1,1,1], "VALID")
        self.dW2 = tf.compat.v1.nn.conv2d_backprop_filter(tf.pad(self.a1,[[0,0],[1,1],[1,1],[0,0]]), self.w[1].shape, self.d2, [1,1,1,1], "VALID")
        self.dW1 = tf.compat.v1.nn.conv2d_backprop_filter(self.x, self.w[0].shape, self.d1, [1,1,1,1], "VALID") #pad not required for x as x is already in 25*25*5

        regcons = eta*0 

        self.w = self.w - regcons*self.w 
        self.b = self.b - regcons*self.b

        self.w[12] = self.w[12] - eta*self.dW13/N 
        self.b[12] = self.b[12] - eta*np.sum(self.d13, 0)/N
        self.w[11] = self.w[11] - eta*self.dW12/N 
        self.b[11] = self.b[11] - eta*np.sum(self.d12, 0)/N
        self.w[10] = self.w[10] - eta*self.dW11/N 
        self.b[10] = self.b[10] - eta*np.sum(self.d11, 0)/N
        self.w[9] = self.w[9] - eta*self.dW10/N 
        self.b[9] = self.b[9] - eta*np.sum(self.d10, 0)/N
        self.w[8] = self.w[8] - eta*self.dW9/N 
        self.b[8] = self.b[8] - eta*np.sum(self.d9, 0)/N
        self.w[7] = self.w[7] - eta*self.dW8/N 
        self.b[7] = self.b[7] - eta*np.sum(self.d8, 0)/N
        self.w[6] = self.w[6] - eta*self.dW7/N 
        self.b[6] = self.b[6] - eta*np.sum(self.d7, 0)/N
        self.w[5] = self.w[5] - eta*self.dW6/N 
        self.b[5] = self.b[5] - eta*np.sum(self.d6, 0)/N
        self.w[4] = self.w[4] - eta*self.dW5/N 
        self.b[4] = self.b[4] - eta*np.sum(self.d5, 0)/N
        self.w[3] = self.w[3] - eta*self.dW4/N 
        self.b[3] = self.b[3] - eta*np.sum(self.d4, 0)/N
        self.w[2] = self.w[2] - eta*self.dW3/N 
        self.b[2] = self.b[2] - eta*np.sum(self.d3, 0)/N
        self.w[1] = self.w[1] - eta*self.dW2/N 
        self.b[1] = self.b[1] - eta*np.sum(self.d2, 0)/N
        self.w[0] = self.w[0] - eta*self.dW1/N 
        self.b[0] = self.b[0] - eta*np.sum(self.d1, 0)/N

        #for i in range(self.w.shape[0]):
        #    self.w[i] = symmetric_w(self.w[i])
        #    self.b[i] = symmetric_b(self.b[i])
        #symmetric weights 

def weight_reset(w_file="w_conv_pr"):
    dat = np.load(w_file + ".npz", allow_pickle=True)
    w = dat['w']
    b = dat['b'] 
    np.savez("w_RL.npz", w=w, b=b)
    print("reset weights") 
    # 需要复制的时候用tf.expand_dims和tf.tile [N,1,1,1] 

def train_dat_reset(t_file = "train_dat_RL.npy"):
    tarr = np.array([[0,0,np.inf,0]]) 
    np.save(t_file, tarr)
    print("reset training rounds")

def board_to_in(board, turn, libs, hist1, hist2,hist3,hist4,hist5,hist6,hist7,hist8, terr):
    c = np.expand_dims(np.tile(np.expand_dims(np.ones(turn.shape), (1,2)), [19,19]), 3)
    s = np.zeros(board.shape)
    o = np.zeros(board.shape)
    ls1 = np.zeros(board.shape)
    ls2 = np.zeros(board.shape)
    ls3 = np.zeros(board.shape)
    ls4 = np.zeros(board.shape)
    ls5 = np.zeros(board.shape)
    ls6 = np.zeros(board.shape)
    ls7 = np.zeros(board.shape)
    ls8 = np.zeros(board.shape)
    lo1 = np.zeros(board.shape)
    lo2 = np.zeros(board.shape)
    lo3 = np.zeros(board.shape)
    lo4 = np.zeros(board.shape)
    lo5 = np.zeros(board.shape)
    lo6 = np.zeros(board.shape)
    lo7 = np.zeros(board.shape)
    lo8 = np.zeros(board.shape)
    e = np.zeros(board.shape)
    for i in range(turn.shape[0]):
        board1 = board[i]*(2*turn[i]-1) 
        lib1 = libs[i]*(2*turn[i]-1) 
        s[i] = np.array(board1 == 1, dtype = int) 
        o[i] = np.array(board1 == -1, dtype = int)
        e[i] = np.array(board1 == 0, dtype = int) 
        ls0 = np.array(lib1 > 0, dtype = int)
        lo0 = np.array(lib1 < 0, dtype = int)*-1 
        ls1[i] = np.array(ls0 == 1, dtype = int) 
        ls2[i] = np.array(ls0 == 2, dtype = int) 
        ls3[i] = np.array(ls0 == 3, dtype = int) 
        ls4[i] = np.array(ls0 == 4, dtype = int) 
        ls5[i] = np.array(ls0 == 5, dtype = int) 
        ls6[i] = np.array(ls0 == 6, dtype = int) 
        ls7[i] = np.array(ls0 == 7, dtype = int)  
        ls8[i] = np.array(ls0 >= 8, dtype = int) 
        lo1[i] = np.array(lo0 == 1, dtype = int) 
        lo2[i] = np.array(lo0 == 2, dtype = int) 
        lo3[i] = np.array(lo0 == 3, dtype = int) 
        lo4[i] = np.array(lo0 == 4, dtype = int) 
        lo5[i] = np.array(lo0 == 5, dtype = int) 
        lo6[i] = np.array(lo0 == 6, dtype = int) 
        lo7[i] = np.array(lo0 == 7, dtype = int) 
        lo8[i] = np.array(lo0 >= 8, dtype = int) 
        terr0 = np.array(terr == 2*turn-1, dtype = int) 
        terr1 = np.array(terr == 1-2*turn, dtype = int) 
    s = tf.pad(np.expand_dims(s, 3),[[0,0],[2,2],[2,2],[0,0]])
    o = tf.pad(np.expand_dims(o, 3),[[0,0],[2,2],[2,2],[0,0]])
    e = tf.pad(np.expand_dims(e, 3),[[0,0],[2,2],[2,2],[0,0]]) 
    c = tf.pad(c,[[0,0],[2,2],[2,2],[0,0]])
    ls1 = tf.pad(np.expand_dims(ls1, 3),[[0,0],[2,2],[2,2],[0,0]])
    ls2 = tf.pad(np.expand_dims(ls2, 3),[[0,0],[2,2],[2,2],[0,0]])
    ls3 = tf.pad(np.expand_dims(ls3, 3),[[0,0],[2,2],[2,2],[0,0]])
    ls4 = tf.pad(np.expand_dims(ls4, 3),[[0,0],[2,2],[2,2],[0,0]])
    ls5 = tf.pad(np.expand_dims(ls5, 3),[[0,0],[2,2],[2,2],[0,0]])
    ls6 = tf.pad(np.expand_dims(ls6, 3),[[0,0],[2,2],[2,2],[0,0]])
    ls7 = tf.pad(np.expand_dims(ls7, 3),[[0,0],[2,2],[2,2],[0,0]])
    ls8 = tf.pad(np.expand_dims(ls8, 3),[[0,0],[2,2],[2,2],[0,0]])
    lo1 = tf.pad(np.expand_dims(lo1, 3),[[0,0],[2,2],[2,2],[0,0]])
    lo2 = tf.pad(np.expand_dims(lo2, 3),[[0,0],[2,2],[2,2],[0,0]])
    lo3 = tf.pad(np.expand_dims(lo3, 3),[[0,0],[2,2],[2,2],[0,0]])
    lo4 = tf.pad(np.expand_dims(lo4, 3),[[0,0],[2,2],[2,2],[0,0]])
    lo5 = tf.pad(np.expand_dims(lo5, 3),[[0,0],[2,2],[2,2],[0,0]])
    lo6 = tf.pad(np.expand_dims(lo6, 3),[[0,0],[2,2],[2,2],[0,0]])
    lo7 = tf.pad(np.expand_dims(lo7, 3),[[0,0],[2,2],[2,2],[0,0]])
    lo8 = tf.pad(np.expand_dims(lo8, 3),[[0,0],[2,2],[2,2],[0,0]])
    h1 = tf.pad(np.expand_dims(hist1, 3),[[0,0],[2,2],[2,2],[0,0]])
    h2 = tf.pad(np.expand_dims(hist2, 3),[[0,0],[2,2],[2,2],[0,0]])
    h3 = tf.pad(np.expand_dims(hist3, 3),[[0,0],[2,2],[2,2],[0,0]])
    h4 = tf.pad(np.expand_dims(hist4, 3),[[0,0],[2,2],[2,2],[0,0]])
    h5 = tf.pad(np.expand_dims(hist5, 3),[[0,0],[2,2],[2,2],[0,0]])
    h6 = tf.pad(np.expand_dims(hist6, 3),[[0,0],[2,2],[2,2],[0,0]])
    h7 = tf.pad(np.expand_dims(hist7, 3),[[0,0],[2,2],[2,2],[0,0]])
    h8 = tf.pad(np.expand_dims(hist8, 3),[[0,0],[2,2],[2,2],[0,0]]) 
    terr0 = tf.pad(terr0,[[0,0],[2,2],[2,2],[0,0]]) 
    terr1 = tf.pad(terr1,[[0,0],[2,2],[2,2],[0,0]]) 

    return tf.cast(np.concatenate((s,o,e,c,ls1,ls2,ls3,ls4,ls5,ls6,ls7,ls8,lo1,lo2,lo3,lo4,lo5,lo6,lo7,lo8,h1,h2,h3,h4,h5,h6,h7,h8, terr0, terr1), 3), dtype = tf.float32)

def remake_board(hists, wls, samp): 
    dim = samp.reshape(-1).shape[0] 
    board_dat = np.zeros([dim, 19, 19])
    libs = np.zeros([dim, 19, 19])
    board_hist1 = np.zeros([dim, 19, 19]) 
    board_hist2 = np.zeros([dim, 19, 19]) 
    board_hist3 = np.zeros([dim, 19, 19]) 
    board_hist4 = np.zeros([dim, 19, 19]) 
    board_hist5 = np.zeros([dim, 19, 19]) 
    board_hist6 = np.zeros([dim, 19, 19])  
    board_hist7 = np.zeros([dim, 19, 19]) 
    board_hist8 = np.zeros([dim, 19, 19]) 
    terr = np.zeros([dim, 19, 19, 8]) 

    board_dat1 = np.zeros([dim, 19, 19])
    board_turn = np.zeros(dim)

    i1 = 0 
    for i in range(samp.shape[0]): 
        this_arr = samp[i]
        k = 0 
        remake = Go()
        for j in range(hists[i].shape[0]):
            if this_arr[k] == j: 
                board_dat[i1] = deepcopy(remake.board) 
                board_turn[i1] = deepcopy(remake.turn)
                l0 = calc_liberties(deepcopy(remake.board))
                libs[i1] = l0.num_liberties*deepcopy(remake.board)
                terr[i1] = e_territory(deepcopy(remake.board)) 
                if j >= 1: 
                    x = np.min([18, np.max([0, hists[i][j-1][0]-1])]) 
                    y = np.min([18, np.max([0, hists[i][j-1][1]-1])]) 
                    board_hist1[i1, x, y] = 1 
                if j >= 2: 
                    x = np.min([18, np.max([0, hists[i][j-2][0]-1])]) 
                    y = np.min([18, np.max([0, hists[i][j-2][1]-1])]) 
                    board_hist2[i1, x, y] = 1     
                if j >= 3: 
                    x = np.min([18, np.max([0, hists[i][j-3][0]-1])]) 
                    y = np.min([18, np.max([0, hists[i][j-3][1]-1])]) 
                    board_hist3[i1, x, y] = 1 
                if j >= 4: 
                    x = np.min([18, np.max([0, hists[i][j-4][0]-1])]) 
                    y = np.min([18, np.max([0, hists[i][j-4][1]-1])]) 
                    board_hist4[i1, x, y] = 1             
                if j >= 5: 
                    x = np.min([18, np.max([0, hists[i][j-5][0]-1])]) 
                    y = np.min([18, np.max([0, hists[i][j-5][1]-1])]) 
                    board_hist5[i1, x, y] = 1 
                if j >= 6: 
                    x = np.min([18, np.max([0, hists[i][j-6][0]-1])]) 
                    y = np.min([18, np.max([0, hists[i][j-6][1]-1])]) 
                    board_hist6[i1, x, y] = 1     
                if j >= 7: 
                    x = np.min([18, np.max([0, hists[i][j-7][0]-1])]) 
                    y = np.min([18, np.max([0, hists[i][j-7][1]-1])]) 
                    board_hist7[i1, x, y] = 1 
                if j >= 8: 
                    x = np.min([18, np.max([0, hists[i][j-8][0]-1])]) 
                    y = np.min([18, np.max([0, hists[i][j-8][1]-1])]) 
                    board_hist8[i1, x, y] = 1          
                if np.all(hists[i][j] == [-1, -1]): 
                    pass 
                else: 
                    x = np.min([18, np.max([0, hists[i][j][0]-1])])
                    y = np.min([18, np.max([0, hists[i][j][1]-1])])
                    board_dat1[i1, x, y] = 1
                i1 += 1 
                k += 1
                if k == samp[i].shape[0]:
                    break
            remake.place(hists[i][j])
        
        del(remake) 
        gc.collect()
    
    bd = board_to_in(board_dat, board_turn, libs, board_hist1,board_hist2,board_hist3,
                     board_hist4,board_hist5,board_hist6,board_hist7,board_hist8, terr) 

    """
    ti = np.random.randint(1, 8)
    if ti == 2:
        bd = np.flip(bd, axis = 1)
        board_dat1 = np.flip(board_dat1, axis = 1)
    if ti == 3:
        bd = np.flip(bd, axis = 2)
        board_dat1 = np.flip(board_dat1, axis = 2)
    if ti == 4:
        bd = np.flip(bd, axis = (1,2))
        board_dat1 = np.flip(board_dat1, axis = (1,2))
    if ti >= 5: 
        bd = tf.transpose(bd, [0, 2, 1, 3])
        board_dat1 = tf.transpose(board_dat1, [0, 2, 1]) 
    if ti == 6:
        bd = np.flip(bd, axis = 1)
        board_dat1 = np.flip(board_dat1, axis = 1)
    if ti == 7:
        bd = np.flip(bd, axis = 2)
        board_dat1 = np.flip(board_dat1, axis = 2)
    if ti == 8:
        bd = np.flip(bd, axis = (1,2))
        board_dat1 = np.flip(board_dat1, axis = (1,2))
    """

    board_dat1 = tf.cast(tf.reshape(board_dat1, [dim, 361]), tf.float32)

    return bd, board_dat1 

class Supervised_Learning(): 
    def __init__(self): 
        self.load_t() 
        self.conv = Conv() 
        self.k = int(self.dat[-1][1]+1)
        self.epoch = deepcopy(self.dat[-1][0])
        self.n_k = len(os.listdir("prof_dat_1/"))

    def Run(self): 
        di = "prof_dat_1/"
        if self.k >= self.n_k-1:
            self.k = 1
            epoch_add = True
        else:
            epoch_add = False 
            self.k += 1
        if epoch_add: 
            self.epoch = self.epoch + 1 
        #print("epoch {}, batch {}, loading data...".format(int(self.epoch)+1, self.k), end = "\r") 
        input_dat = np.load(di + "sample_{}.npz".format(self.k), allow_pickle = True) 
        self.inX = tf.cast(tf.pad(input_dat['inX'], [[0,0],[2,2],[2,2],[0,0]]), tf.float32) 
        self.y = np.reshape(input_dat['y'], [-1, 361]).astype(int) 
        self.wl = input_dat['wl']
        for i in range(self.wl.shape[0]):
            if self.wl[i] == -1: 
                self.y[i] = (1-self.y[i])/360
        #print("epoch {}, batch {}, training...    ".format(int(self.epoch)+1, self.k), end = "\r")
        self.conv.backward(self.inX, self.y)
        self.MSE = self.conv.loss
        self.acc = tf.reduce_sum(tf.multiply(self.y, self.conv.p)) 
        #self.conv.loss_calc(self.inX, self.y) 
        #self.MSE_expost = self.conv.loss
        self.MSE = self.MSE/self.y.shape[0] 
        self.acc = self.acc/self.y.shape[0] 
        #self.MSE_expost = self.MSE_expost/self.y.shape[0]
        self.sum_prob = np.sum(self.conv.p)
        self.pos_prob = np.all(self.conv.p > 0)

    def load_t(self, t_file = "train_dat_RL.npy"):
        self.dat = np.load(t_file) 

    def save_t(self, t_file = "train_dat_RL.npy"):
        dat1 = np.array([[self.epoch, self.k, self.MSE, self.acc]])
        self.dat = np.append(self.dat, dat1, axis = 0)
        np.save(t_file, self.dat) 

if __name__ == '__main__': 
    if reset == True: 
        weight_reset()
        train_dat_reset() 
    if new_round == True: 
        train_dat_reset() 
    a = Supervised_Learning() #t loaded 
    for i in range(30000):
        if a.epoch < 200: 
            a.Run()
        else: 
            break 
            print("Training Finished") 
        if np.mod(a.k, 10) == 0:
            norm_gradient_1 = np.average(np.abs(a.conv.dW1))
            norm_size_1 = np.average(np.abs(a.conv.w[0]))
            norm_size_13 = np.average(np.abs(a.conv.w[12]))
            norm_gradient_13 = np.average(np.abs(a.conv.dW13))
            print("epoch {}, batch {}, sum_prob {:.2e}, pos_pob {}, A {:.2e}, Z {:.2e}".format(int(a.epoch+1), a.k, a.sum_prob, a.pos_prob, norm_size_1, norm_size_13))
        if np.mod(a.k, 100) == 0: 
            if a.MSE < 7:
                a.conv.save_w()
                a.save_t() 
        if np.mod(a.epoch, 10) == 9 and np.mod(a.k, 100) == 0:
            if a.MSE < 7:
                w_f = "backup_pol/w e" 
                t_f = "backup_pol/t e" 
                a.conv.save_w(w_file = w_f + str(int(a.epoch)) + "b" + str(a.k) + "round 0")
                a.save_t(t_file = t_f + str(int(a.epoch))+"b"+str(a.k)+"round 0"+".npy")
                #print("weights backed up")
            else: 
                print("weight explosion warning, retraining decade")
                a.conv.load()  
                a.load_t() 
        if a.k == 2 and a.epoch >= 1: 
            input_sampling.sample() 
            print("resampled") 
