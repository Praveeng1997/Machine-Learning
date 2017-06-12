import pdb
class loan:
    def __init__(self,o,s,i,default):
       self.owner = o  #1 for yes
       self.status = s #1 for marrried,2 for divorced,0 for single
       self.income = i 
       self.default = default #1 for yes


class Assignment:
    n1 = []
    n2 = []
    n1count1 = 0
    n2count1 = 0
    n1count2 = 0
    n2count2 = 0


own = Assignment()
mart = Assignment()
inc = Assignment()

F = [own,mart,inc]


class node:
    def __init__(self,train):
        self.lst = train
        self.left = None
        self.right = None
        self.cond = []
        self.label = None
        




martial = [' Single ','Married ','Divorced']
status = ['No','Yes ']
l = []
l.append(loan(1,0,125,0))
l.append(loan(0,1,100,0))
l.append(loan(0,0,70,0))
l.append(loan(1,1,120,0))
l.append(loan(0,2,95,1))
l.append(loan(0,1,60,0))
l.append(loan(1,2,220,0))
l.append(loan(0,0,85,1))
l.append(loan(0,1,75,0))
l.append(loan(0,0,90,1))

print "Tid Home Martial Income Defaulted\n"
for i in range(len(l)):
    print str(i+1)+" "+(status[l[i].owner])+" "+(martial[l[i].status])+" "+str((l[i].income))    +" "+(status[l[i].default])

def gini(count1,count2):
    count =count1+count2
    Gini = 1 -((count1/count)*(count1/count))-((count2/count)*(count2/count))
    return Gini

def weight_gini(m):
    count = m.n1count1+m.n1count2+m.n2count1+m.n2count2
    g1 = ((m.n1count1+m.n1count2)/float(count))*gini(m.n1count1,m.n1count2)
    g2 =  ((m.n2count1+m.n2count2)/float(count))*gini(m.n2count1,m.n2count2)
    return g1+g2


def income_split(l):
    I = []
    for i in range(len(l)):
        I.append(l[i].income)
    I.sort()
    sp = [55]
    for i in range(len(I)-1):
        sp.append((I[i]+I[i+1])//2)
    sp.append(230)
    print sp
    return sp
    




def gini_parent(l):
    count1=0.0
    count2=0.0
    count = len(l)
    for i in range(len(l)):
        if(l[i].default == 0):
            count1 += 1
        else:
            count2 +=1
    Gini = 1 - ((count1**2+count2**2))/(float(count**2))
    return Gini
                                                            
        
        



def assign_nodes(l):
    pdb.set_trace() 
    for i in range(len(l)):
        if l[i].owner == 1 :
            own.n1.append(l[i])
            s = own.n1[own.n1count1+own.n1count2]
            if(s.default == 0):
                own.n1count1 += 1
            else :
                own.n1count2 += 1
        else :
             own.n2.append(l[i])
             s = own.n2[own.n2count1+own.n2count2]
             if(s.default == 0):
                own.n2count1 += 1
             else :
                own.n2count2 += 1

   # Assign Status

        if ( l[i].status == 2 or l[i].status == 0) :
            mart.n1.append(l[i])
            s = mart.n1[mart.n1count1+mart.n1count2]
            if(s.default == 0):
                mart.n1count1 += 1
            else :
                mart.n1count2 += 1
        else :
             mart.n2.append(l[i])
             s = mart.n2[mart.n2count1+mart.n2count2]
             if(s.default == 0):
                mart.n2count1 += 1
             else :
                mart.n2count2 += 1
'''
   # Assgin Income
def assign_income(l):
    S = income_split(l)
    for i in range(len(l)):
        if(l[i].inc <= S[i]):
            inc.n1.append(l[i])
            s = inc.n1[inc.n1count1+inc.n1count2]
            if(s.default == 0):
                inc.n1count1 +=1
             else :
                inc.n1count2 += 1
        else :
             inc.n2.append(l[i])
             s = inc.n2[inc.n2count1+inc.n2count2]
             if(s.default == 0):
                inc.n2count1 += 1
             else :
                inc.n2count2 += 1
                
'''





print "Income\n"
income_split(l)

                
assign_nodes(l)
print "\n House Owner"
print "N1(Yes):"+str(own.n1count1)+","+str(own.n1count2)
print "N2(No):"+str(own.n2count1)+","+str(own.n2count2)
for i in range(len(own.n1)):
    print own.n1[i].income
print '---------------'

print "\n Martial Status"
print "N1(Single,Divorced):"+str(mart.n1count1)+","+str(mart.n1count2)
print "N2(Married):"+str(mart.n2count1)+","+str(mart.n2count2)
    
#print gini_parent(l)
                                                                      
'''def best_split(l,F):
    parent_gini = gini_parent(l)
    wgini_o = weight_gini(F[0])
    gain1 = parent_gini - wgini_o
    print gain1
    wgini_m = weight_gini(F[1])
    gain2 = parent_gini - wgini_m
    print gain2
   # wgini_i = weight_gini(F[2])
    wgini_i = 0.9
    gain3 = parent_gini - wgini_i
    print gain3
    Gain = max(gain1,gain2,gain3)
    if(Gain == gain1): best = F[0]
    elif(Gain == gain2):best = F[1]
    else:best = F[2]
    
    return best

'''



def best_split(l,F):
    parent_gini = gini_parent(l)
    wgini_o = weight_gini(F[0])
   # gain1 = parent_gini - wgini_o
    print wgini_o
    wgini_m = weight_gini(F[1])
   # gain2 = parent_gini - wgini_m
    print wgini_m
   # wgini_i = weight_gini(F[2])
    wgini_i = 0.4
  #  gain3 = parent_gini - wgini_i
    print wgini_i
    Gain = max(wgini_o,wgini_m,wgini_i)
    if(Gain == wgini_o): best = F[0]
    elif(Gain == wgini_m):best = F[1]
    else:best = F[2]
    
    return best











    
#best_split(own,mart,inc)

def stopping_condition(Node,F):
    status = True
    for i in (Node.lst):
        if(i.default != 0):
            status = False
            break
    for i in (Node.lst):
        status = True
        if(i.default != 1):
           status = False
           break
    return status

def classify(node):
    t = 0
    f = 0
    for items in (node.lst):
        if(items.default ==0):
            t +=1
        else :
            f +=1
    if(t >= f):
        status = True
    else: status = False
    return status

#Root Node


first = node([])
for i in range(len(l)):
    first.lst.append(l[i])
for i in range(len(l)):
    print first.lst[i].income
first.label = classify(first)
first_cond = best_split(first.lst,F)
first.cond = first_cond











# Tree Growth Algorithm
def TreeGrowth(Node):
#    pdb.set_trace()
   
    Node.label = classify(Node)
    if(stopping_condition(Node,F)==True):
        print "Stopped"
        Node.cond = [-1,1]
        return Node
    else :
        
        cond=best_split(Node.lst,F)
        print "Best Split"
        print cond.n1count1
        Node.cond = cond

        child1 = node([])
        child2 = node([])

        for i in range(len(Node.cond.n1)):
            print Node.cond.n1[i].default
        
        child1.lst.extend(cond.n1)
        for i in range(len(child1.lst)):
            print child1.lst[i].income
        child2.lst.extend(cond.n2)
            
            
        child1.cond = best_split(child1.lst,F)
        child2.cond = best_split(child2.lst,F)
        

        Node.left = TreeGrowth(child1)
        Node.right = TreeGrowth(child2)

        child1.label = classify(child1,F)
        child2.label = classify(child2,F)

        Node.label = classify(Node)

        return Node
    
first = TreeGrowth(first)
