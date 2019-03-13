import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import random

def classification_error(w, X, y):
    bestErr = len(X)
    err_cnt = 0
    N = len(X)
    for n in range(N):
        s = np.sign(w.T.dot(X[n])) # if this is zero, then :(
        if y[n] != s:
            err_cnt += 1
    if err_cnt < bestErr:
        bestErr = err_cnt
    print(err_cnt)
    return err_cnt
    return bestErr

def choose_miscl_point(w, X, y):
    mispts = []
    # Choose a random point among the misclassified
    for n in range(len(X)):
        if np.sign(w.T.dot(X[n])) != y[n]:
            mispts.append((X[n], y[n]))
    #print(len(mispts))
    return mispts[random.randrange(0,len(mispts))]

def plotAlg(X,y,w):
    #plots data
    c0 = plt.scatter(X[y==-1,0],X[y==-1,1],s=20,color='r', marker='x' )
    c1 = plt.scatter(X[y==1,0],X[y==1,1], s=20, color='b' , marker='o' )
    
    #plot hypothesis
    m, b = -w[1]/w[2], -w[0]/w[2]
    l = np.linspace(min(X[:,1]),max(X[:,1]))
    plt.plot(l, m*l+b, 'k-')
    
    #displays legend
    plt.axis(option="auto")
    plt.legend((c0,c1),(' All_Other_Numbers_-1' , 'Number_Zero_+1') ,
        loc = 'upper_right' , scatterpoints=1, fontsize=11)
    # displays axis legends and title
    plt.xlabel(r'$x_1$' )
    plt.ylabel(r'$x_2$')
    plt.title(r'Intensity_and_Symmetry_of_Digits')
    # saves the figure into a .pdf file (desired!)
    #plt.savefig('midterm.plot.pdf' , bbox_inches='tight')
    plt.show()

#Linear Regression

# read digits data & split it into X (training input) and y (target output)
dataset = genfromtxt('C:\\Users\\Cameron\\Documents\\DATA440\\midterm\\features.csv', delimiter=' ' )
y = dataset[:, 0]
X = dataset[:, 1 : ]
y[y!=4] = -1 #rest of numbers are negative class
y[y==4] = +1 #number four is the positive class

#linear regression
Xo = np.append(np.ones((len(X),1)), X, 1)   # add a column of ones
Xs = np.linalg.pinv(Xo.T.dot(Xo)).dot(Xo.T)
wlr = Xs.dot(y)

plotAlg(X, y, wlr)#Initial Solution

it = 0 

stopIt = 50
currIt = 0
bestW = wlr
bestE = len(X)
# Iterate until all points are correctly classified
nerr = classification_error(wlr, Xo, y)
while nerr != 0:
    nerr = classification_error(wlr, Xo, y)
    it += 1
    currIt += 1
    if currIt > stopIt:
        print("Early stop! no progress")
        break
    # Pick random misclassified point
    x, s = choose_miscl_point(wlr, Xo, y)
    # Update weights
    wlr += s*x
    nerr = classification_error(wlr, Xo, y)
    if nerr < bestE:
        currIt=0
        bestE = nerr
        bestW = wlr
print("Now for the pocket algorithm")
#starting pocket algorithm at bestW
w = bestW


stopIt = 50
currIt = 0
bestW = w
bestE = len(X)
# Iterate until all points are correctly classified
nerr = classification_error(w, Xo, y)
while nerr != 0:
    nerr = classification_error(w, Xo, y)
    it += 1
    currIt += 1
    if currIt > stopIt:
        print("Early stop! no progress")
        break
    # Pick random misclassified point
    x, s = choose_miscl_point(w, Xo, y)
    # Update weights
    w += s*x
    nerr = classification_error(w, Xo, y)
    if nerr < bestE:
        currIt=0
        bestE = nerr
        bestW = w
plotAlg(Xo, y, bestW)
print("The number of iterations is: " + str(it))
print("The best error classification is: " + str(bestE))