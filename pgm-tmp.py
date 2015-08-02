__author__ = 'BeRo'
import numpy as np
import pandas as pd
import math
from scipy import optimize as optimize
from numpy.linalg import norm
import corenlp
import string
from  time import time

'''########################################Variables#######################################'''
Train_Set = []
Language_Model = []
Dictionary = []
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
def Get_Probability(x, Dic):
    if x in Dic.keys():
        return Dic[x]
    else:
        return -50
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
    char = '#'
    for k in range(1, len(x)):
        begin = x[:k]    # from beginning to n (n not included)
        end = x[k:]    # n+1 through end of string
        tmp_string = begin + char + end
        P_List.append(tmp_string)
    return P_List
#------------------------------------------------------------------------------------------
def Build_Y(X, index, zi, O):

    x = X.split(" ")
    Z = []
    o = []
    Z.append(zi)
    o.append("Nothing")

    if O == "Spell":
        tmp = Deletion(zi)
        for i in tmp:
            Z.append(i)
            o.append("Deletion")
        tmp = Insertion(zi)
        for i in tmp:
            Z.append(i)
            o.append("Insertion")
        tmp = Substitution(zi)
        for i in tmp:
            Z.append(i)
            o.append("Substitution")
        tmp = Transposition(zi)
        for i in tmp:
            Z.append(i)
            o.append("Transposition")

    elif O == "Split":
        tmp = Splitting(zi)
        for i in tmp:
            Z.append(i)
            o.append("Splitting")

    elif O == "Merg":
        if index < len(x)-1:
            tmp = zi +'-'+ x[index+1]
            tmp = tmp.split("*")
            for i in tmp:
                Z.append(i)
                o.append("Merging")

    return [Z, o]
#------------------------------------------------------------------------------------------
def isword(x, Dic):
    if abs(x.find('#') - x.find('-')) == 1:
        return 0
    dash_point = x.find('-')
    if dash_point != -1:
        first = x[:dash_point]    # from beginning to n (n not included)
        secound = x[dash_point+1:]    # n+1 through end of string
        sharp_point = first.find('#')
        if sharp_point != -1:
            begin = first[:sharp_point]
            end = first[sharp_point+1:]
            tmp_str = begin + end
            if Get_Probability(tmp_str, Dic) < -49:
                return 0
        else:
            if Get_Probability(first, Dic) < -49:
                return 0
        sharp_point = secound.find('#')
        if sharp_point != -1:
            begin = secound[:sharp_point]
            end = secound[sharp_point+1:]
            tmp_str = begin + end
            if Get_Probability(tmp_str, Dic) < -49:
                return 0
        else:
            if Get_Probability(secound, Dic) < -49:
                return 0
    sharp_point = x.find('#')
    if sharp_point != -1:
        begin = x[:sharp_point]
        end = x[sharp_point+1:]
        tmp_str = begin + end
        if Get_Probability(tmp_str, Dic) < -49:
            return 0
    else:
        if Get_Probability(x, Dic) < -49:
            return 0
    return 1
#------------------------------------------------------------------------------------------
def Get_Possible_Y(X, index, Dic):
    x = X.split(" ")

    zi = x[index]
    Y = []
    O = []

    '''Spell Split Merg'''
    '''
    [Z, o] = Build_Y(X, index, zi, "Spell")
    for i in range(len(Z)):
        [ZZ, oo] = Build_Y(X, index, Z[i], "Split")
        for j in range(len(ZZ)):

            [ZZZ, ooo] = Build_Y(X, index, ZZ[j], "Merg")
            for k in range(len(ZZZ)):
                Y.append([Z[i], ZZ[j], ZZZ[k]])
                O.append([o[i], oo[j], ooo[k]])
    '''


#    '''Spell Merg Split'''
#    [Z, o] = Build_Y(X, index, zi, "Spell")
#    for i in range(len(Z)):
#        [ZZ, oo] = Build_Y(X, index, Z[i], "Merg")
#        for j in range(len(ZZ)):
#            [ZZZ, ooo] = Build_Y(X, index, ZZ[j], "Split")
#            for k in range(len(ZZZ)):
#                if isword(ZZZ[k], Dic) == 1 or (o[i] == "Nothing" and oo[j] == "Nothing" and ooo[k] == "Nothing"):
#                    Y.append([Z[i], ZZ[j], ZZZ[k]])
#                    O.append([o[i], oo[j], ooo[k]])

    '''Split Spell Merg'''
    '''
    [Z, o] = Build_Y(X, index, zi, "Split")
    for i in range(len(Z)):
        [ZZ, oo] = Build_Y(X, index, Z[i], "Spell")
        for j in range(len(ZZ)):
            [ZZZ, ooo] = Build_Y(X, index, ZZ[j], "Merg")
            for k in range(len(ZZZ)):
                Y.append([Z[i], ZZ[j], ZZZ[k]])
                O.append([o[i], oo[j], ooo[k]])
    '''

    '''Split Merg Spell'''
    '''
    [Z, o] = Build_Y(X, index, zi, "Split")
    for i in range(len(Z)):
        [ZZ, oo] = Build_Y(X, index, Z[i], "Merg")
        for j in range(len(ZZ)):
            [ZZZ, ooo] = Build_Y(X, index, ZZ[j], "Spell")
            for k in range(len(ZZZ)):
                Y.append([Z[i], ZZ[j], ZZZ[k]])
                O.append([o[i], oo[j], ooo[k]])
    '''

    '''Merg Split Spell'''
    [Z, o] = Build_Y(X, index, zi, "Merg")
    for i in range(len(Z)):
        [ZZ, oo] = Build_Y(X, index, Z[i], "Split")
        for j in range(len(ZZ)):
            [ZZZ, ooo] = Build_Y(X, index, ZZ[j], "Spell")
            for k in range(len(ZZZ)):
                if isword(ZZZ[k], Dic) == 1 or (o[i] == "Nothing" and oo[j] == "Nothing" and ooo[k] == "Nothing"):
                    Y.append([Z[i], ZZ[j], ZZZ[k]])
                    O.append([o[i], oo[j], ooo[k]])



#    '''Merg Spell Split'''
#    [Z, o] = Build_Y(X, index, zi, "Merg")
#    for i in range(len(Z)):
#        [ZZ, oo] = Build_Y(X, index, Z[i], "Spell")
#        for j in range(len(ZZ)):
#            [ZZZ, ooo] = Build_Y(X, index, ZZ[j], "Split")
#            for k in range(len(ZZZ)):
#                if isword(ZZZ[k], Dic) == 1 or (o[i] == "Nothing" and oo[j] == "Nothing" and ooo[k] == "Nothing"):
#                    Y.append([Z[i], ZZ[j], ZZZ[k]])
#                    O.append([o[i], oo[j], ooo[k]])

    return [Y, O]
#------------------------------------------------------------------------------------------
def Test_Model(X, lamda, lm, Dic):    #X is the input string, lamda is the parameter
    x = X.split(" ")
    Best_O = []
    Best_y = []

    yi_1 = "<s>"
    index = 0
    isreplace = 0
    merged_before = 0

    for i in  range(len(x)):
        Best_P = -10000
        Best_oi = ""
        Best_yi = ""

        if merged_before == 1:
            index += 1
            continue

        [Y, O] = Get_Possible_Y(X, i, Dic)

        for yi in range (len(Y)):
            zi = Y[yi][len(Y[yi])-1]
            if zi.find('-') != -1:
                begin = zi[:zi.find('-')]    # from beginning to n (n not included)
                end = zi[zi.find('-')+1:]    # n+1 through end of string
                zi = begin
            if zi.find('#') != -1:
                fp = zi[:zi.find('#')]    # from beginning to n (n not included)
                sp = zi[zi.find('#')+1:]    # n+1 through end of string
                tmp_string = fp + sp
            else:
                tmp_string = zi
            score = lamda * f(yi_1, tmp_string, lm) + lamda #+ lamda *Get_Probability(tmp_string, lm)#* h(yi, oi, X, index)

            if score > Best_P:
                Best_oi = O[yi]
                Best_zi = Y[yi]
                Best_yi = zi
                Best_P = score

        isreplace = 0
        merged_before = 0
        tmp_varialbe = Best_zi[len(Best_zi)-1]
        if tmp_varialbe.find('-') != -1:
            fp = zi[:tmp_varialbe.find('-')]    # from beginning to n (n not included)
            sp = zi[tmp_varialbe.find('-')+1:]    # n+1 through end of string
            if sp.find('#') != -1:
                first = sp[:sp.find('#')]
                sec = sp[sp.find('#')+1,:]
                next_y = first+sec
                isreplace = 1
            else:
                next_y = sp
                isreplace = 0
        elif tmp_varialbe.find('#') != -1:
            merged_before = 1

        index += 1
        Best_O.append(Best_oi)
        Best_y.append(Best_yi)
        yi_1 = Best_yi

    return Best_O, Best_y
#------------------------------------------------------------------------------------------
def F_Function(lamda):
    Train_Data = Train_Set
    lm = Language_Model
    sum = 0
    for i in range(1, len(Train_Data)):
        Y = Train_Data.ix[i, 1]
        y = Y.split(" ")
        yi_1 = "<s>"
        for j in range(len(y)):
            yi = y[j]
            score =  lamda * f(yi_1, yi, lm) + lamda * 1 #h(yi, oi, X, j))
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
        Train_Set.append([currentline[0], currentline[1]])

    train = pd.DataFrame(Train_Set)
    Train_Set = train
    myfile.close()

    Language_Model = pd.DataFrame().from_csv("bigram.csv")
    Language_Model.columns = ["digram", "log prob"]
    Language_Model = to_dict(Language_Model)

    Dictionary = pd.DataFrame().from_csv("onegram.csv")
    Dictionary.columns = ["digram", "log prob"]
    Dictionary = to_dict(Dictionary)

    #Get_Possible_Y("ho w to mace catt", 1, Language_Model)
    landa = optimize.minimize(F_Function, [1], method='L-BFGS-B', bounds = ((0, None),) ,options=dict({'maxiter':10}))

    start = time()
    O_landa = landa.x[0]
    print "best landa is: " , O_landa
    #O_landa = 0.000793534564046
    x = Test_Model("ho w to mace catt", O_landa, Language_Model, Dictionary)
    print(x)
    print("--------------finish-----------------")
    print (time()-start)