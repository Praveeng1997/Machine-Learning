import random
import pdb
import math
import cPickle
import numpy
import matplotlib.pyplot as plt
import copy
import collections
from  scipy import misc
from PIL import Image
import gzip
import skimage.measure
import scipy.ndimage
import scipy.signal

def oz(x):
    if x>0.3:
        return 1
    else :
        return 0
nip = 2
inp =[]
op=[]
cin = []
cop = []
#inp = numpy.array([[0,0],[0,1],[1,1],[1,0]])
#op = numpy.array([[0],[1],[1],[1]])
counthist = 0

mnisto = cPickle.load(open("opdataset.p","rb"))

nip = 100
for i in range(nip):
    cin.append(train_set[0][i])
    cop.append(mnisto[i])
cin = numpy.asarray(cin)
cop = numpy.asarray(cop)


def csig(x):
    return 1/float((1+math.exp(-x)))

def sigmoid(temp):
    temp = temp.tolist()
    for i in range(len(temp)):
        for j in range(len(temp[i])):
            if temp[i][j] < 20 and temp[i][j] > -20:
                temp[i][j] = 1/float((1+math.exp(-temp[i][j])))
            elif temp[i][j] > 0:
                temp[i][j] = 1
            else :
                temp[i][j] = 0
        
    return numpy.array(temp)
class hidden:
    def __init__(self,i,n,o,h):
        self.no_nodes = n
        self.no_input = i
        self.no_output = o
        self.h = numpy.zeros((h,n))
        self.ho = numpy.zeros((h,n))
        self.delta = self.ho
        self.der = self.delta
        self.inwght = numpy.random.uniform(low = -0.5,high = 0.5,size =(i,n))
        self.inwghttrans = self.inwght.transpose()
        self.deltatrans = self.delta.transpose()
        self.hotrans = self.ho.transpose()
        self.momentum = numpy.zeros((i,n))
        self.delin = self.momentum
        self.sumdelinsqr = self.delin
        self.v = self.momentum
class conv:
    global mr,mc
    def __init__(self,r,c,m):
        self.row = r
        self.col = c
        self.mask = numpy.random.uniform(-0.5,0.5,size=(depth,nip,mr,mc))
        a,b,c = m[0].shape
        self.deltam = numpy.zeros((depth,a,b,c))
        self.mat = m
        self.conv = numpy.zeros((depth,nip,r-mr+1,c-mc+1))
        self.relu = numpy.zeros((depth,nip,r-mr+1,c-mc+1))
        self.deltac = numpy.zeros((depth,nip,r-mr+1,c-mc+1))
        if (r-mr+1)%2 == 0 :
            self.pool = numpy.zeros((depth,nip,(r-mr+1)/2,(c-mc+1)/2))
        else :
            self.pool = numpy.zeros((depth,nip,((r-mr+1)/2)+1,((c-mc+1)/2)+1))
        self.delta = numpy.zeros((self.pool.shape))
        self.delma =numpy.zeros((depth,nip,mr,mc))
        self.der =numpy.zeros((depth,nip,mr,mc))
layer = []
hid = [500,50]
def create_layers(numex,inp,out):
    global hid
    layer.append(hidden(0,inp,hid[0],numex))
    if len(hid) != 1:
        for i in range(len(hid)):
            h = hid
            if i == 0:
                layer.append(hidden(layer[0].no_nodes,h[i],h[i+1],numex))
                continue
            elif i == len(hid)-1:
                layer.append(hidden(h[i-1],h[i],out,numex))
                continue
            else : 
                layer.append(hidden(h[i-1],h[i],h[i+1],numex))
    else :
        layer.append(hidden(inp,hid[0],out,numex))
    layer.append(hidden(layer[-1].no_nodes,out,0,numex))
#pdb.set_trace()

def backpropagation(layer):
    check = 1
    count = 1
    plotx = []
    ploty = []
    while count < 2 :
        #----------------forwardfeed------------------
        layer[0].ho = layer[0].h = inp
	layer[0].der = layer[0].ho*(numpy.ones(layer[0].ho.shape)-layer[0].ho)
        for i in range(len(layer)-1):
            layer[i+1].h = numpy.dot(layer[i].ho,layer[i+1].inwght)
            layer[i+1].ho = sigmoid(layer[i+1].h)
            layer[i+1].der = layer[i+1].ho*(numpy.ones(layer[i+1].ho.shape)-layer[i+1].ho)
        
        #---------backwardfeed---------------
        one = numpy.ones(layer[-1].ho.shape)
        layer[-1].delta=numpy.subtract(op,layer[-1].ho)*layer[-1].ho*(numpy.subtract(one,layer[-1].ho))
        layer[-1].delta *= 0.01
        error = numpy.square(numpy.subtract(op,layer[-1].ho))*0.5
        check = numpy.mean(error)
        for i in range(len(layer)-2,-1,-1):
            layer[i+1].deltatrans = layer[i+1].delta.transpose()
            layer[i+1].hotrans = layer[i+1].ho.transpose()
            layer[i].delta = numpy.dot(layer[i+1].inwght,layer[i+1].deltatrans).transpose()
            layer[i].delta *= layer[i].der*0.01
            layer[i+1].delin = numpy.dot(layer[i].ho.transpose(),layer[i+1].delta)
            layer[i+1].inwght += (layer[i+1].delin + layer[i+1].momentum)
        count += 1
    return check


cinp = []
for i in range(len(cin)):
    cinp.append(cin[i].reshape((28,28)))
    for j in range(28):
        for k in range(28):
            cinp[i][j][k] = (cinp[i][j][k])
cinp = numpy.asarray(cinp)


mr = 3
mc = 3
pf = 2
nr,nc = cinp[0].shape
r = nr
inc  = copy.deepcopy(cinp)
depth = 20
convlayer = []
incc = []
for i in range(depth):
    incc.append(inc)
convlayer.append(conv(nr,nc,incc))
cl = 0
while nr>6 and nc > 6:
    opc = numpy.zeros((depth,nip,nr-mr+1,nc-mc+1))

    #pdb.set_trace()
    incc = []
    for dpt in range(depth):
        for tc in range(nip):
            opc[dpt][tc] = scipy.signal.convolve2d(convlayer[cl].mat[dpt][tc],convlayer[cl].mask[dpt][tc],mode='valid')
            convlayer[cl].relu[dpt][tc] = numpy.maximum(opc[dpt][tc],0)
            convlayer[cl].conv[dpt][tc] = opc[dpt][tc]
            convlayer[cl].pool[dpt][tc]=skimage.measure.block_reduce(convlayer[cl].relu[dpt][tc],(2,2),numpy.mean)
        nr,nc = convlayer[cl].pool[dpt][0].shape
        inc = copy.deepcopy(convlayer[cl].pool[dpt])
        incc.append(inc)
    convlayer.append(conv(nr,nc,incc))
    cl+=1

#pdb.set_trace()
print nip
inp = numpy.zeros((nip,depth*(convlayer[-2].pool[0].shape[1])*(convlayer[-2].pool[0].shape[2])))


count = 0
create_layers(nip*depth,len(inp[0]),10)
op = copy.deepcopy(cop)
while True:
    batch = 100
    error = 100
    cin = []
    cop = []
    for i in range(nip):
        cin.append(train_set[0][i])
        cop.append(mnisto[i])
    cin = numpy.asarray(cin)
    cop = numpy.asarray(cop)
    op = copy.deepcopy(cop)
    while batch<20000:
        nr,nc = cinp[0].shape
        for cl in range(len(convlayer)):
            for dpt in range(depth):
                opc = numpy.zeros((depth,nip,nr-mr+1,nc-mc+1))
                for tc in range(nip):
                    #pdb.set_trace()
                    opc[dpt][tc] = scipy.signal.convolve2d(convlayer[cl].mat[dpt][tc],numpy.flip(numpy.flip(convlayer[cl].mask[dpt][tc],0),1),mode='valid')
                    #opc[dpt][tc] = scipy.signal.convolve2d(convlayer[cl].mat[dpt][tc],(convlayer[cl].mask[dpt][tc]),mode='valid')
                    convlayer[cl].relu[dpt][tc] = numpy.maximum(opc[dpt][tc],0)
                    convlayer[cl].conv[dpt][tc] = opc[dpt][tc]
                    convlayer[cl].pool[dpt][tc]=skimage.measure.block_reduce(convlayer[cl].relu[dpt][tc],(2,2),numpy.max)
                    if cl != len(convlayer)-1 :
                        convlayer[cl+1].mat[dpt][tc]=convlayer[cl].pool[dpt][tc]
            nr,nc = convlayer[cl].pool[dpt][0].shape
            #pdb.set_trace()
            #func = numpy.vectorize(oz)
        for i in range(nip):
            inc=[]
            for j in range(depth):
                inc.append(convlayer[-1].mat[j][i].flatten())
            inc = numpy.asarray(inc)
            inc =inc.flatten()
            inp[i] = inc
        
        #pdb.set_trace()
        error = backpropagation(layer)
        #pdb.set_trace()
        for i in range(nip):
            for j in range(depth):
                #convlayer[-1].deltam[i] = layer[0].delta[i].reshape((convlayer[-2].delta[i].shape))
                a,b=layer[0].delta.shape
                convlayer[-1].deltam[j][i] = layer[0].delta[i][j*b/depth:(j+1)*b/depth].reshape((convlayer[-2].delta[j][i].shape))
        #print convlayer[-2].delta
        #pdb.set_trace()
        for cl in range(len(convlayer)-2,-1,-1):
            for dpt in range(depth):
                convlayer[cl].delta = copy.deepcopy(convlayer[cl+1].deltam)
                for i in range(nip):
                    temp1 = scipy.ndimage.zoom(convlayer[cl].pool[dpt][i],2,order=0)
                    temp2 = scipy.ndimage.zoom(convlayer[cl].delta[dpt][i],2,order=0)
                    if len(convlayer[cl].relu[dpt][i])%2 != 0:
                        temp1 = numpy.delete(temp1,-1,axis=0)
                        temp1 = numpy.delete(temp1,-1,axis=1)
                        temp2 = numpy.delete(temp2,-1,axis=0)
                        temp2 = numpy.delete(temp2,-1,axis=1)
                    #pdb.set_trace()
                    convlayer[cl].deltac[dpt][i] = temp2*(temp1 == convlayer[cl].relu[dpt][i])*1
                    #convlayer[cl].deltam[i] = scipy.signal.convolve2d(convlayer[cl].deltac[i],(convlayer[cl].mask[i]),mode='full')
                    convlayer[cl].deltam[dpt][i] = scipy.signal.convolve2d(convlayer[cl].deltac[dpt][i],numpy.flip(numpy.flip(convlayer[cl].mask[dpt][i],0),1),mode='full')
                    #convlayer[cl].delma[dpt][i] = scipy.signal.convolve2d(convlayer[cl].mat[dpt][i],(convlayer[cl].deltac[dpt][i]),mode='valid')
                    convlayer[cl].delma[dpt][i] = scipy.signal.convolve2d(numpy.flip(numpy.flip(convlayer[cl].deltac[dpt][i],0),1),(convlayer[cl].mat[dpt][i]),mode='valid')
                    convlayer[cl].mask[dpt][i] += convlayer[cl].delma[dpt][i]
        if batch != 50000:
            batch += 100
        else :
            batch = 100
        del(cin)
        del(cop)
        cin = []
        cop = []
        for i in range(batch,batch+100,1):
            cin.append(train_set[0][i])
            cop.append(mnisto[i])
        cin = numpy.asarray(cin)
        cop = numpy.asarray(cop)
        op = copy.deepcopy(cop)
        #print error
        #print batch
        #print convlayer[1].mask[0][0]
        #print convlayer[1].mask[1][0]
        #print convlayer[1].mask[2][0]
        #print layer[-1].ho[1]
        #print op[1]
        #print layer[-1].ho[99]
        #print op[99]
        #pdb.set_trace()
    print error
    #print count
        
    cin=[]
    cop=[]
    for i in range(nip):
        cin.append(test_set[0][i])
        cop.append(mnistto[i])
    cin = numpy.asarray(cin)
    cop = numpy.asarray(cop)
    op = copy.deepcopy(cop)
    nr,nc = cinp[0].shape
    batch = 0
    summm = 0.0
    while batch<5000:
        nr,nc = cinp[0].shape
        for cl in range(len(convlayer)):
            for dpt in range(depth):
                opc = numpy.zeros((depth,nip,nr-mr+1,nc-mc+1))
                for tc in range(nip):
                    #pdb.set_trace()
                    opc[dpt][tc] = scipy.signal.convolve2d(convlayer[cl].mat[dpt][tc],numpy.flip(numpy.flip(convlayer[cl].mask[dpt][tc],0),1),mode='valid')
                    #opc[dpt][tc] = scipy.signal.convolve2d(convlayer[cl].mat[dpt][tc],(convlayer[cl].mask[dpt][tc]),mode='valid')
                    convlayer[cl].relu[dpt][tc] = numpy.maximum(opc[dpt][tc],0)
                    convlayer[cl].conv[dpt][tc] = opc[dpt][tc]
                    convlayer[cl].pool[dpt][tc]=skimage.measure.block_reduce(convlayer[cl].relu[dpt][tc],(2,2),numpy.max)
                    if cl != len(convlayer)-1 :
                        convlayer[cl+1].mat[dpt][tc]=convlayer[cl].pool[dpt][tc]
            nr,nc = convlayer[cl].pool[dpt][0].shape
        
        #pdb.set_trace()
        error = backpropagation(layer)
    
        func = numpy.vectorize(oz)
        #trueop = func(layer[-1].ho)
        trueop = copy.deepcopy(layer[-1].ho)
        for i in range(len(trueop)):
            m = max(trueop[i])
            for j in range(len(trueop[i])):
                if trueop[i][j]==m:
                    trueop[i][j]=1
                else :
                    trueop[i][j] = 0
        for i in range(len(op)):
            summm += 1*(numpy.all(trueop[i]==op[i]))       
        if batch != 10000:
            batch += 100
        else :
            batch = 100
        del(cin)
        del(cop)
        cin = []
        cop = []
        for i in range(batch,batch+100,1):
            cin.append(train_set[0][i])
            cop.append(mnisto[i])
        cin = numpy.asarray(cin)
        cop = numpy.asarray(cop)
        op = copy.deepcopy(cop)
        #print error
        #print batch
    print count
    print summm/5000.0
    print '---------------------------'
    count += 1
