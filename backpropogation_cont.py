import random
import math
import pdb

class training_network:
    def __init__(self,inputs):
        self.lists = inputs
        
#pdb.set_trace()

t = []        
t.append(training_network([0,0,0]))
t.append(training_network([0.6,-1,0.4]))
t.append(training_network([1,0.251,0.8]))
t.append(training_network([1,0,1]))

for i in range(len(t)) :
    print t[i].lists


def initialize_weights(n1,n2):
    weight = [random.uniform(-0.05,0.05) for i in range(n1*n2)]
    return weight

def compute_hidden(w,i):
    h1 = w[0]*i[0]+w[1]*i[1]
    h2 = w[2]*i[0]+w[3]*i[1]
    h3 = w[4]*i[0]+w[5]*i[1]
    hidden = [h1,h2,h3]
    return hidden

def compute_output(h,w):
    output = h[0]*w[0]+h[1]*w[1]+h[2]*w[2]
    return output

def squash(x):
    y = 1+math.exp(-x)
    return (1/float(y))
# To initialie nodes and weights

class nodes:
    def __init__(self,data,ni,nh,no):
        self.inputs = [0 for i in range(ni)]
        self.hidden = [0 for i in range(nh)]
        self.out = [0 for i in range(no)]
        
        for i in range(len(self.inputs)):
           self.inputs[i]=data[i]
           
        for i in range(len(self.out)):
            self.out[i] = data[-1]
            

 #Hidden layer
        self.weight_h = initialize_weights(ni,nh)
        self.hidden_values = compute_hidden(self.weight_h,self.inputs)
        for i in range(nh):
            self.hidden[i] = squash(self.hidden_values[i])

 #Output
 
        self.weight_o = initialize_weights(nh,no)
        self.out1 = compute_output(self.hidden,self.weight_o)
        self.output = squash(self.out1)

def calculate_error(n):
    E = 0.5*((n.out[0] - n.output)**2)
    return E

n = []
for i in range(len(t)):
    n.append(nodes(t[i].lists,2,3,1))
    print str(n[i].inputs[0])+","+str(n[i].inputs[1])
    print n[i].out[0]
    print str(n[i].hidden[0])+","+str(n[i].hidden[1])+","+str(n[i].hidden[2])

print "Weights of Hidden and Output\n"
for i in range(len(t)):
    print n[i].weight_h
    print n[i].weight_o
    print "\n"

print "Obtained Outputs\n"
for i in range(len(t)):
    print str(n[i].output)+"\n"

E = []
err_count =0
def Backpropogate (n,rate,ni,nh,no):
    err_count = 1
    count =0
   
    print "Initial Error\n"
    for k in range(len(n)):
        E.append(calculate_error(n[k]))
        print E[k]
    while (err_count > 0.001) :
        count += 1
        for i in range(len(n)):
           # Error for Output
           out = n[i].output
           delta_o = out*(1-out)*(out-n[i].out[0])
           delta_h = []
            # Error for Hidden
           for j in range(nh):
               h = 0
               h = n[i].hidden[j]*(1-n[i].hidden[j])*delta_o*n[i].weight_o[j]
               delta_h.append(h)
               
           #Update Weights
           
           for j in range(nh*no):
               d_w = 0
               d_w = -rate*delta_o*(n[i].hidden[j])
               n[i].weight_o[j] += d_w

           for j in range(nh):
                d_ho = -rate*delta_h[j]*(n[i].inputs[0])
                d_h1 = -rate*delta_h[j]*(n[i].inputs[1])
                n[i].weight_h[2*j] += d_ho
                n[i].weight_h[(2*j)+1] += d_h1
           if count%100000 ==0:
               print E[i]
           hidden_values = compute_hidden(n[i].weight_h,n[i].inputs)
           for m in range(nh):
               n[i].hidden[m] = squash(n[i].hidden_values[m])

                   
           n[i].out1 = compute_output(n[i].hidden,n[i].weight_o)
           n[i].output = squash(n[i].out1)
           E[i] = calculate_error(n[i])
        err_count = max(E)
    return count
           
count = Backpropogate(n,0.5,2,3,1)

print "\n Final Error\n"
for i in range(len(n)):
    print E[i]


print "\n Final Outputs\n"
for i in range(len(n)):
    print str(n[i].output)+"\n"

print "\n Total Count:"+str(count)
