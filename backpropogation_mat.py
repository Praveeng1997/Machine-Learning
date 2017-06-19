import random
import numpy as np
import pdb
import math


class layers:
    def __init__(self,size1,size2):
        self.matrix = np.zeros((size1,size2))

class weight:
    def __init__(self,slayer1,slayer2):
        self.matrix = np.random.uniform(low = -0.5,high =0.5,size = (slayer1,slayer2))

inp = layers(4,2)
h = []
h_d = []
t = layers(4,1)
#h.append(layers(4,2))
h.append(layers(4,3))
out = layers(4,1)

w = []


w.append(weight(inp.matrix.shape[1],h[0].matrix.shape[1]))
for i in range(len(h)):
    if i < len(h)-1:
        w.append(weight(h[i].matrix.shape[1],h[i+1].matrix.shape[1]))
w.append(weight(h[-1].matrix.shape[1],out.matrix.shape[1]))

#pdb.set_trace()
# Target Output

t.matrix[0][0] = 0
t.matrix[1][0] = 1
t.matrix[2][0] = 0
t.matrix[3][0] = 1


#print "Size of H:"+str(len(h))
#print "Size of W:"+str(len(w))

# Initialize Inputs

inp.matrix[0][0],inp.matrix[0][1] = 0,0
inp.matrix[1][0],inp.matrix[1][1] = 0,1
inp.matrix[2][0],inp.matrix[2][1] = 1,1
inp.matrix[3][0],inp.matrix[3][1] = 1,0



def squash(m):
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            s = 1+math.exp(-m[i][j])
            m[i][j] = (1/float(s))
    return m

def initialize():
    l1 = multiply(inp.matrix,w[0].matrix)
    h[0].matrix = squash(l1)
    
    for i in range(len(h)):
        if i < len(h)-1:
            l = multiply(h[i].matrix,w[i+1].matrix)
            h[i+1].matrix = squash(l)

    lo = multiply(h[-1].matrix,w[-1].matrix)
    out.matrix = squash(lo)



# Multiply Matrices

def multiply(m1,m2):
    product = np.dot(m1,m2)
    return product

# Compute Derivatives

def derivative(m,d):
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            d[i][j] = m[i][j]*(1-m[i][j])
    return d



def print_():
    
    print "Input"+str(inp.matrix)
    print "\n Hidden"
    for i in range(len(h)):
        print h[i].matrix
        print "\n"
    print "\n Weights"
    for i in range(len(w)):
        print w[i].matrix
        print "\n"
    print "\n"+str(out.matrix)
initialize()

rate =0.75
n = 0

def Backpropogate():
    #pdb.set_trace()
    n = 0
    stop = False
    
    while stop!=True :
       # pdb.set_trace()
        
        #print_()
        diff = t.matrix-out.matrix
        n +=1
        d_o = np.zeros((out.matrix.shape[0],out.matrix.shape[1]))
        d_o = derivative(out.matrix,d_o)
        h_d = []
        for i in range(len(h)):
            h1 = h[i].matrix*(np.ones(h[i].matrix.shape)-(h[i].matrix))
            h_d.append(h1)
                  
        delta_o = rate*diff*d_o
        
        Delta = []
        Delta.append(delta_o)
        for i in range(len(h)):
          #  D = rate*h_d[len(h)-i-1] *(np.dot(Delta[i],w[len(h)-i].matrix.transpose()))


            D =(np.dot(w[len(h)-i].matrix,Delta[i].transpose())).transpose()
            D *= rate*h_d[len(h)-i-1]
            Delta.append(D)
            
        D_W = []

       
        D_W.append(np.dot(inp.matrix.transpose(),Delta[-1]))

        for i in range(len(h)):
              
           D_W.append(np.dot(h[i].matrix.transpose(),Delta[len(h)-i-1]))
        #   pdb.set_trace()    
        for i in range(len(w)):
          
            w[i].matrix +=D_W[i]
        initialize()
        E = 0.5 * diff *diff
        if (max(E) < 0.001):
            stop = True
        print out.matrix
    return n    
             
n = Backpropogate()

print n
