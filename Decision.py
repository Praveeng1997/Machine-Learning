import graphviz as gv

g1 = gv.Graph(format='svg')
class person:
    def __init__(self,h,m,i,d,ids):
        self.cond = []
        self.cond.append(h);
        self.cond.append(m);
        self.cond.append(i)
        self.ids = ids
        self.db = d

marital = ['single  ','married ','divorced']
truth = ['No ','Yes']

class node:
    def __init__(self,train):
        self.lst = train
        self.left = None
        self.right = None
        self.cond = []
    def best_label(self):
        t1 = 0
        t2 = 0
        for items in self.lst:
            if items.db == 0:
                t1 += 1
            elif items.db == 1:
                t2 += 1
        if (t1>t2):
            self.label = 'give'
        else :
            self.label = 'dont give'
n = 10

train_set = []

train_set.append(person(1,0,125,0,0))
train_set.append(person(0,1,100,0,1))
train_set.append(person(0,0,70,0,2))
train_set.append(person(1,1,120,0,3))
train_set.append(person(0,2,95,1,4))
train_set.append(person(0,1,60,0,5))
train_set.append(person(1,2,220,0,6))
train_set.append(person(0,0,85,1,7))
train_set.append(person(0,1,75,0,8))
train_set.append(person(0,0,90,1,9))

ntc = 3

status =[0 for i in range(ntc)] 

#function to find the best split condition of the input train
def best_test_cond(train):
    gini = []
    nxt = [0 for i in range(n)]
    for i in range(ntc):
        gini.append(0)
        if status[i] == 0:
            temp = [train[j].cond[i] for j in range(len(train))]
            uniq = set(temp)
            uniq = list(uniq)
            uniq.sort()
            diff = 0
            r = 0
            if len(uniq) > 1 :
                for j in range(len(uniq)-1):
                    chunk1 = uniq[0:j+1]
                    chunk2 = uniq[j+1:len(temp)]
                    c1 = 0
                    c2 = 0
                    c1g = 0
                    c2g = 0
                    tg = 0
                    ##########################################################
                    for k in range(len(train)):
                        if train[k].db == 0:
                            tg += 1
                            if train[k].cond[i] in chunk1:
                                c1 += 1#number of elements in chunk1
                                c1g += 1#number of elemnts of chunk1 having class 0  
                            elif train[k].cond[i] in chunk2:
                                c2 += 1# n0 in 2
                                c2g += 1
                            else :
                                print 'error'


                        elif train[k].db == 1:
                            if train[k].cond[i] in chunk1:
                                c1 += 1#number of elements in chunk1
                            elif train[k].cond[i] in chunk2:
                                c2 += 1# n0 in 2
                            else :
                                print 'error'
                        else :
                            print 'error'
                    ###########################################################
                    ginic1 = 1 - (c1g**2 + (len(chunk1)-c1g)**2)/(len(chunk1)**2)#gini of chunk1
                    ginic2 = 1 - (c2g**2 + (len(chunk2)-c2g)**2)/(len(chunk2)**2)#gini of chunk2
                    ginit = 1 - (tg**2 + (len(temp)-tg)**2)/(len(temp)**2)#gini of main
                    gini12 = (len(chunk1)*ginic1 + len(chunk2)*ginic2)/len(temp)
                    if diff < (ginit-gini12):
                        diff = ginit - gini12
                        r = j
                nxt[i] = j
                gini[i] = diff
    retgini = max(gini)
    retcond = gini.index(retgini)
    condchunk = nxt[retcond]
    retlist = []
    retlist.append(retcond)
    retlist.append(condchunk)
    #print retlist
    return(retlist)

#print all the train sets
'''
for i in range(n):
                print str(i+1) + '  ' + truth[train_set[i].cond[0]] +'  '+ marital[train_set[i].cond[1]] + '  ' + str(train_set[i].cond[2]) + 'k  ' + truth[train_set[i].db]
'''
von = 0
#function to return a string for plotting a graph
def string(zzz,ppp):
    global von
    a = {}
    a[0] = 'home owner'
    a[1] = 'marital status'
    a[2] = 'salary'
    b = []
    b.append(['no','yes'])
    b.append(['single','married','divorced'])
    b.append(['60','70','75','85','90','95','100','120','125','220'])
    if zzz[0] != -1:
        s =   ' ' + str(b[zzz[0]][zzz[1]]) +'  ' +  str(a[zzz[0]]) +'  '+  str(b[zzz[0]][zzz[1]+1])
    else :
        s = ppp
    von += 1
    return s
#function to return the stopping condition of the decision tree

def stop_cond(nod):
    train = nod.lst
    condlist = nod.cond
    temp = [train[j].db for j in range(len(train))]
    uniq = set(temp)
    uniq = list(uniq)
    if len(uniq) == 1:
        return True
    temp = [train[j].cond[condlist[0]] for j in range(len(train))]
    uniq = set(temp)
    uniq = list(uniq)
    chunk1 = uniq[0:condlist[1] + 1]
    chunk2 = uniq[condlist[1]+1:len(uniq)]
   

  #  print temp
  #  print condlist
  #  print chunk1
  #  print chunk2
    
    if len(chunk1) == 0:
        return True
    if len(chunk2) == 0:
        return True
    return False

    
#create first node add all train sets find its best label 
first = node([])
for i in range(len(train_set)):
    first.lst.append(person(train_set[i].cond[0],train_set[i].cond[1],train_set[i].cond[2],train_set[i].db,train_set[i].ids))
first.best_label()
first_cond = best_test_cond(first.lst)
first.cond = first_cond
g1.node(string(first.cond,first.label))


def treegrowth(nod):
    nod.best_label()
    if stop_cond(nod):
        nod.cond = [-1,-1]
        return nod
    else :
        cond = best_test_cond(nod.lst)
        nod.cond = cond 
        temp = [nod.lst[i].cond[cond[0]] for i in range(len(nod.lst))]
        uniq = set(temp)
        uniq = list(uniq)
        chunk1 = uniq[0:cond[1]+1]
        chunk2 = uniq[cond[1]+1:len(nod.lst)]

        
        child1 = node([])
        child2 = node([])

        for i in range(len(nod.lst)):
            if nod.lst[i].cond[cond[0]] in chunk1 :
                child1.lst.append(nod.lst[i])
            else :
                child2.lst.append(nod.lst[i])
                
        child1.cond = best_test_cond(child1.lst)
        child2.cond = best_test_cond(child2.lst)
        
        nod.left = treegrowth(child1)
        nod.right = treegrowth(child2)

        child1.best_label();
        child2.best_label();
        nod.best_label();
        g1.node(string(child2.cond,child2.label))
        g1.node(string(child1.cond,child1.label))
        g1.edge(string(nod.cond,nod.label),string(child1.cond,child1.label),color = 'red')
        g1.edge(string(nod.cond,nod.label),string(child2.cond,child2.label),color = 'green')
        ''' 
        for x in nod.lst:
        print x.ids 
        print '--------------' 
'''
        return nod
                
#print stop_cond(first)
first = treegrowth(first)

#print first.left.left.left.cond
'''
g1.node('a')
g1.node('b')
g1.edge('a','b')
'''
g1.render(filename = 'img/g1')

