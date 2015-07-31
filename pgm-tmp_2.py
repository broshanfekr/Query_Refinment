__author__ = 'BeRo'
import numpy as np
import pandas as pd
import math
from scipy import optimize as optimize
from numpy.linalg import norm
import corenlp
import string

'''########################################Variables#######################################'''
Train_Set = []
Language_Model = []
###########################################################################################
'''#######################################Functions########################################'''
def to_dict(lm_df):
    return lm_df.set_index('digram').to_dict()['log prob']
#------------------------------------------------------------------------------------------
def f(yi_1, yi, lm):
    yi_1 = yi_1.lower()
    yi = yi.lower()
    tmp = yi_1 + ' ' + yi
    if tmp  in lm.keys():
        x = lm[tmp]
        return lm[tmp]
    else:
        return -100
#------------------------------------------------------------------------------------------
def h(yi, oi, x, i):# All inputs are string exept i. i is integer
    yi = yi.lower()
    x = x.lower()
    if oi == "Nothing":
        return 1
    elif oi == "Deletion":   #Spelling Correction-> Deletion
        myx = x.split(" ")
        myx = myx[i]
        for k in range(len(myx)):
            begin = myx[:k]    # from beginning to n (n not included)
            end = myx[k+1:]    # n+1 through end of string
            tmp_string = begin + end
            if yi == tmp_string:
                return 1
        return 0
    elif oi == "Stemming":
        myx = x.split(" ")
        myx = myx[i]
        list = ["ing", "ed", "s", "es"]
        for k in list:
            tmp_string = myx + k
            if yi == tmp_string:
                return 1
        return 0
    elif oi == "Insertion":
        myx = x.split(" ")
        myx = myx[i]
        list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                'u', 'v', 'w', 'x', 'y', 'z']
        for k in range(len(myx)+1):
            for char in list:
                begin = myx[:k]    # from beginning to n (n not included)
                end = myx[k:]    # n+1 through end of string
                tmp_string = begin + char + end
                if yi == tmp_string:
                    return 1
        return 0
    elif oi == "Substitution":
        myx = x.split(" ")
        myx = myx[i]
        list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                'u', 'v', 'w', 'x', 'y', 'z']
        for k in range(len(myx)):
            for char in list:
                begin = myx[:k]    # from beginning to n (n not included)
                end = myx[k+1:]    # n+1 through end of string
                tmp_string = begin + char + end
                if yi == tmp_string:
                    return 1
        return 0
    elif oi == "Transposition":
        myx = x.split(" ")
        myx = myx[i]
        t = []
        for char in myx:
            t.append(char)
        myx = t
        for k1 in range(len(myx)):
            for k2 in range(k1+1, len(myx)):
                tmp = myx[k1]
                myx[k1] = myx[k2]
                myx[k2] = tmp
                tmp_string = ""
                for kk in range(len(myx)):
                    tmp_string = tmp_string + myx[kk]
                if yi == tmp_string:
                    return 1
        return 0
    elif oi == "Splitting":
        myx = x.split(" ")
        myx = myx[i]
        char = '-'
        for k in range(1, len(myx)):
            begin = myx[:k]    # from beginning to n (n not included)
            end = myx[k:]    # n+1 through end of string
            tmp_string = begin + char + end
            if yi == tmp_string:
                return 1
        return 0
    elif oi == "Merging":
        myx = x.split(" ")
        myxi = myx[i]
        myxii = myx[i+1]
        tmp_string = myxi + "-" + myxii
        if yi == tmp_string:
            return 1
        return 0
    elif oi == "Merged_Before":
        print("do nothing")
#------------------------------------------------------------------------------------------
def Deletion(x):
    P_List = []
    for k in range(len(x)):
        begin = x[:k]    # from beginning to n (n not included)
        end = x[k+1:]    # n+1 through end of string
        tmp_string = begin + end
        P_List.append(tmp_string)
    return P_List
def Stemming(x):
    P_List = []
    list = ["ing", "ed", "s", "es"]
    for k in list:
        tmp_string = x + k
        P_List.append(tmp_string)
    return P_List
def Insertion(x):
    P_List = []
    list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
            'u', 'v', 'w', 'x', 'y', 'z']
    for k in range(len(x)+1):
        for char in list:
            begin = x[:k]    # from beginning to n (n not included)
            end = x[k:]    # n+1 through end of string
            tmp_string = begin + char + end
            P_List.append(tmp_string)
    return P_List
def Substitution(x):
    P_List = []
    list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
            'u', 'v', 'w', 'x', 'y', 'z']
    for k in range(len(x)):
        for char in list:
            begin = x[:k]    # from beginning to n (n not included)
            end = x[k+1:]    # n+1 through end of string
            tmp_string = begin + char + end
            P_List.append(tmp_string)
    return P_List
def Transposition(x):
    P_List = []
    t = []
    for char in x:
        t.append(char)
    x = t
    for k1 in range(len(x)):
        for k2 in range(k1+1, len(x)):
            tmp = x[k1]
            x[k1] = x[k2]
            x[k2] = tmp
            tmp_string = ""
            for kk in range(len(x)):
                tmp_string = tmp_string + x[kk]
            P_List.append(tmp_string)

    return P_List
def Splitting(x):
    P_List = []
    char = '-'
    for k in range(1, len(x)):
        begin = x[:k]    # from beginning to n (n not included)
        end = x[k:]    # n+1 through end of string
        tmp_string = begin + char + end
        P_List.append(tmp_string)
    return P_List

#------------------------------------------------------------------------------------------
def Test_Model(X, lamda, lm):    #X is the input string, lamda is the parameter
    O_List = ["Nothing","Splitting", "Merging", "Deletion", "Stemming", "Insertion", "Substitution", "Transposition"]
    x = X.split(" ")
    Best_O = []
    Best_y = []

    yi_1 = "<s>"
    index = 0
    poi = ""
    for i in  range(len(x)):
        Best_P = -10000
        Best_oi = ""
        Best_yi = ""
        if poi == "Merging":
            poi = ""
            index += 1
            continue
        for oi in O_List:
            if oi == "Nothing":
                Y = []
                Y.append(x[i])
                Y.append("</s>")
            if oi == "Deletion":
                Y = Deletion(x[i])
            elif oi == "Stemming":
                Y = Stemming(x[i])
            elif oi == "Insertion":
                Y = Insertion(x[i])
            elif oi == "Substitution":
                Y = Substitution(x[i])
            elif oi == "Transposition":
                Y = Transposition(x[i])
            elif oi == "Splitting":
                Y = Splitting(x[i])
            elif oi ==  "Merging":
                if i < len(x)-1:
                    Y = x[i] +'-'+ x[i+1]
                    Y = Y.split("*")
                else:
                    continue

            for yi in Y:
                if yi == "</s>":
                    break
                if oi == "Splitting":
                    y = yi.split("-")
                    score = (lamda * f(y[0], y[1], lm) + lamda * h(yi, oi, X, index))
                elif oi == "Merging":
                    y = yi.split("-")
                    y = y[0] + y[1]
                    score = (lamda * f(yi_1, y, lm) + lamda * h(yi, oi, X, index))
                else:
                    score = (lamda * f(yi_1, yi, lm) + lamda * h(yi, oi, X, index))
                if score > Best_P:
                    Best_oi = oi
                    Best_yi = yi
                    Best_P = score

        index += 1
        Best_O.append(Best_oi)
        if Best_oi == "Merging":
            tmp = Best_yi.split("-")
            Best_yi = tmp[0]+tmp[1]
        poi = Best_oi
        Best_y.append(Best_yi)
        yi_1 = Best_yi

    return Best_O, Best_y
#------------------------------------------------------------------------------------------
def F_Function(lamda):
    Train_Data = Train_Set
    lm = Language_Model
    sum = 0

    for i in range(1, len(Train_Data)):
        X = Train_Data.ix[i, 0]
        Y = Train_Data.ix[i, 1]
        O = Train_Data.ix[i, 2]
        O = O.strip()
        x = X.split(" ")
        y = Y.split(" ")
        o = O.split(" ")
        yi_1 = "<s>"
#        print(i)
        if i == 18:
            continue
        if i == 60 or i == 61 or i == 64:
            continue
        if i == 100:
            print("suse")
        for j in range(len(x)):
            xi = x[j]
            yi = y[j]
            oi = o[j]
            if yi == '-':
                tmp = yi_1.split("-")
                yi_1 = tmp[0] + tmp[1]
                continue
            #print(lamda)
            if oi == "Merging":
                tmp = yi.split("-")
                score = (lamda * f(yi_1, tmp[0] + tmp[1], lm) + lamda * h(yi, oi, X, j))
            else:
                score = (lamda * f(yi_1, yi, lm) + lamda * h(yi, oi, X, j))
            yi_1 = yi
            sum += score
    nnnn = norm(lamda)
    sum -= norm(lamda)**2 * 10000000
    print "score is: ", sum
    print "landa is: ", lamda
    result= -1 * sum
    return result

'''###########################################################################################'''
if __name__ == "__main__":


    myfile = open("train.csv", "r")
    for line in myfile:
        line = line.strip("\n")
        currentline = line.split(",")
        act = currentline[3]
        act.strip()
        act.split("-")
        mystring = ""
        for s in act:
            if s == "n":
                mystring = mystring + "Nothing "
            elif s == "d":
                mystring = mystring + "Deletion "
            elif s == "i":
                mystring = mystring + "Insertion "
            elif s == "m":
                mystring = mystring + "Merging "
            elif s == "mb":
                mystring = mystring + "Merged_Before "

        Train_Set.append([currentline[1], currentline[2],mystring])
    train = pd.DataFrame(Train_Set)
    Train_Set = train

    myfile.close()

    Language_Model = pd.DataFrame().from_csv("lm.csv")
    Language_Model.columns = ["digram", "log prob"]
    Language_Model = to_dict(Language_Model)

    landa = optimize.minimize(F_Function, [1], method='L-BFGS-B', options=dict({'maxiter':10}))

    O_landa = landa.x[0]
    #O_landa = 1
    x = Test_Model("ho w to mace catt", O_landa, Language_Model)
    print(x)
    print("--------------finish-----------------")