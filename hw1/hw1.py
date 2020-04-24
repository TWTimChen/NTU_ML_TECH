import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib
from libsvm.svmutil import *
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-q', '--question', required=True, help='number of the question to diplay')
args = ap.parse_args()

file_train = "./features.train.txt"
file_test = "./features.test.txt"
x_train = []
y_test = []

with open(file_train, 'r', newline='\n') as f:
    rows = csv.reader(f, delimiter=' ')
    train = [[float(ele) for ele in row if ele!=''] for row in rows]
    
with open(file_test, 'r', newline='\n') as f:
    rows = csv.reader(f, delimiter=' ')
    test = [[float(ele) for ele in row if ele!=''] for row in rows]

train = np.array(train)
test = np.array(train)
x_train = train[:,1:]
x_test = test[:,1:]


###############################################################################
############################        Question 11     ###########################
###############################################################################

if args.question == '11':
    y_train_q11 = [1 if row[0]==0 else -1 for row in train]
    y_test_q11 = [1 if row[0]==0 else -1 for row in test]
    prob  = svm_problem(y_train_q11, x_train)

    def getWeightNorm(c_str):
        
        param = svm_parameter(('-t 0 -c '+c_str))
        m = svm_train(prob, param)
        p_label, p_acc, p_val = svm_predict(y=[1,1,1,1], x=[[0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [0.,0.,1.]], m=m)
        w_q11 = np.array(p_val[1:])
        w_q11 = w_q11-p_val[0]
        return np.linalg.norm(w_q11)

    C = [1e-5, 1e-3, 1e-1, 1e+1, 1e+3]
    C_str = [str(c) for c in C]
    w_norm = [getWeightNorm(c_str) for c_str in C_str]
    plt.figure()
    plt.title("Q11: C verses Length of weight")
    plt.bar(C_str, w_norm)
    plt.show()

###############################################################################
############################        Question 12     ###########################
###############################################################################

if args.question == '12':
    y_train_q12 = [1 if row[0]==8 else -1 for row in train]
    y_test_q12 = [1 if row[0]==8 else -1 for row in test]

    def getEin(c_str):
        prob  = svm_problem(y_train_q12, x_train)
        param = svm_parameter('-t 1 -d 2 -c '+c_str) #Set parameters
        m = svm_train(prob, param) #Training
        p_label, p_acc, p_val = svm_predict(y_train_q12, x_train, m) #Predict training set
        return (100-p_acc[0])/100 #Calculate E_in

    C = [1e-5, 1e-3, 1e-1, 1e+1, 1e+3]
    C_str = [str(c) for c in C]
    Ein_q12 = [getEin(c_str) for c_str in C_str]

    plt.figure()
    plt.title("Q12: C verses E_in")
    plt.bar(C_str, Ein_q12)
    plt.show()

###############################################################################
############################        Question 13     ###########################
###############################################################################

if args.question == '13':
    y_train_q12 = [1 if row[0]==8 else -1 for row in train]
    y_test_q12 = [1 if row[0]==8 else -1 for row in test]

    def getNumSV(c_str):
        prob  = svm_problem(y_train_q12, x_train)
        param = svm_parameter('-t 1 -d 2 -c '+c_str) #Set parameters
        m = svm_train(prob, param) #Training
        return(m.get_nr_sv()) #Get the number of SVs

    C = [1e-5, 1e-3, 1e-1, 1e+1, 1e+3]
    C_str = [str(c) for c in C]
    nSV_q12 = [getNumSV(c_str) for c_str in C_str]

    plt.figure()
    plt.title("Q13: C verses Number of SV")
    plt.bar(C_str, nSV_q12)
    plt.show()

###############################################################################
############################        Question 14     ###########################
###############################################################################

if args.question == '14':
    y_train_q14 = [1 if row[0]==0 else -1 for row in train]
    y_test_q14 = [1 if row[0]==0 else -1 for row in test]

    def K_rbf(x1, x2, g=80):
        return np.exp(-g*sum((x1-x2)**2))

    def getFreeSVDist(c_str, y=y_train_q14):
        prob  = svm_problem(y, x_train)
        param = svm_parameter('-t 2 -g 80 -c '+c_str)
        m = svm_train(prob, param)
        SV = m.get_SV() #Get SVs
        SV = [np.array([sv[1], sv[2]]) for sv in SV] #Extract SV from dictionary to numpy.array
        SV_idx = m.get_sv_indices() #Get the indices of SVs in the training data
        a = m.get_sv_coef() #Get coeficient a
        a = [aa[0] for aa in a] #Extract a from nested list to list
        nSV = len(SV) #Get the number of SVs
        # Calculate the ||w||^2
        w_norm2 = np.sum([a[i]*a[j]*y[SV_idx[i]-1]*y[SV_idx[j]-1]*K_rbf(SV[i], SV[j]) for i in range(nSV) for j in range(nSV)])
        return(w_norm2**(-1/2)) #return 1/||w||

    C = [1e-3, 1e-2, 1e-1, 1e+0, 1e+1]
    C_str = [str(c) for c in C]
    dist_q14 = [getFreeSVDist(c_str) for c_str in C_str]

    plt.figure()
    plt.title("Q14: C verses Distance to Hyperplane")
    plt.bar(C_str, dist_q14)
    plt.show()

###############################################################################
############################        Question 15     ###########################
###############################################################################

if args.question == '15':
    y_train_q14 = [1 if row[0]==0 else -1 for row in train]
    y_test_q14 = [1 if row[0]==0 else -1 for row in test]
    prob  = svm_problem(y_train_q14, x_train)
    param = svm_parameter('-t 2 -g 80 -c 0.1')
    m = svm_train(prob, param)

    def getEout(g_str):
        prob  = svm_problem(y_train_q14, x_train)
        param = svm_parameter('-t 2 -c 0.1 -g '+g_str) #Ser parameters
        m = svm_train(prob, param) #Training
        p_label, p_acc, p_val = svm_predict(y_test_q14, x_test, m) #Predict testing set
        return (100-p_acc[0])/100 #Calculate E_out

    G = [1e0, 1e+1, 1e+2, 1e+3, 1e+4]
    G_str = [str(g) for g in G]
    Eout_q15 = [getEout(g_str) for g_str in G_str]

    plt.figure()
    plt.title("Q15: Gamma verses E_out")
    plt.bar(G_str, Eout_q15)
    plt.show()

###############################################################################
############################        Question 16     ###########################
###############################################################################

if args.question == '16':
    import random
    y_train_q14 = [1 if row[0]==0 else -1 for row in train]
    y_test_q14 = [1 if row[0]==0 else -1 for row in test]
    idx = np.arange(len(x_train)) #Set indices from sampling
    x_train = np.array(x_train)
    y_train_q14 = np.array(y_train_q14)

    # Function to calculate E_out with assigned gamma value
    def getEval(g_str, x_train, y_train, x_val, y_val):
        prob  = svm_problem(y_train, x_train)
        param = svm_parameter('-t 2 -c 0.1 -g '+g_str)
        m = svm_train(prob, param)
        p_label, p_acc, p_val = svm_predict(y_val, x_val, m)
        return (100-p_acc[0])/100

    def getBestG():
        random.shuffle(idx) #Shuffle the indices
        x_val_q16 = x_train[idx[:1000], ] #Get the first 1000 indices for validating set
        y_val_q16 = y_train_q14[idx[:1000]] #Get the first 1000 indices for validating set
        x_train_q16 = x_train[idx[1000:], ] #Get the rest of the indices for training set
        y_train_q16 = y_train_q14[idx[1000:]] # Get the rest of the indices for testing set
        
        G = [1e-1, 1e0, 1e+1, 1e+2, 1e+3]
        G_str = [str(g) for g in G]
        #Calculate E_out for each gamma in G_str
        Eout_q15 = [getEval(g_str, x_train_q16, y_train_q16, x_val_q16, y_val_q16) for g_str in G_str]
        return G_str[np.argmin(Eout_q15)] #Return the gamma with the minimal E_out

    resBestG = [getBestG() for _ in range(100)]

    from collections import Counter
    resCount = dict(Counter(resBestG)) #Count duplicates of each gamma in the result
    G = [1e-1, 1e0, 1e+1, 1e+2, 1e+3]
    G_str = [str(g) for g in G]
    #Fill out the list of gamma's duplicate
    G_choice = [resCount.get(g_str) if resCount.get(g_str) else 0 for g_str in G_str] 

    plt.figure()
    plt.title("Q16: Gamma verses Best Count")
    plt.bar(G_str, G_choice)
    plt.show()
