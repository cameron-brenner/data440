import numpy as np

delta = 0.05 #confidence error
epsilon = 0.05 #generalization error
d_vc = 10 #VC dimension
initGuess = d_vc * 1000 #initial guess for N
nextGuess = 0
count = 1
tempGuess = initGuess

for i in range(1,100):
    #print("tempguessin : " + str(tempGuess)) personal  book keeping 
    print("The initial guess for iteration " + str(count) + " of the generalization is: " +str(tempGuess))
    count+=1
    nextGuess = 8 / epsilon**2 * np.log((4 * ((2 * tempGuess) ** d_vc + 1)) / delta)
    print("The output of this guess is: " + str(nextGuess) + ".")
    if(abs(tempGuess - nextGuess)>500):
        tempGuess = nextGuess
        #print("tempguessout : " + str(tempGuess))    personal book keeping
        print("now for the next iteration!")
    else:
        break
 
    
print("The Guess for sample size with a VC dimension of " + str(d_vc) + "  is: " + str(nextGuess))