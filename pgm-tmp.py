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
def Get_Probability(x, lm):
    if x in lm.keys():
        return lm[x]
    else:
        return -50
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
def Get_Possible_Y(X, index, lm):
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


    '''Spell Merg Split'''
    [Z, o] = Build_Y(X, index, zi, "Spell")
    for i in range(len(Z)):
        [ZZ, oo] = Build_Y(X, index, Z[i], "Merg")
        for j in range(len(ZZ)):
            [ZZZ, ooo] = Build_Y(X, index, ZZ[j], "Split")
            for k in range(len(ZZZ)):
                Y.append([Z[i], ZZ[j], ZZZ[k]])
                O.append([o[i], oo[j], ooo[k]])

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
                Y.append([Z[i], ZZ[j], ZZZ[k]])
                O.append([o[i], oo[j], ooo[k]])

    '''Merg Spell Split'''
    [Z, o] = Build_Y(X, index, zi, "Merg")
    for i in range(len(Z)):
        [ZZ, oo] = Build_Y(X, index, Z[i], "Spell")
        for j in range(len(ZZ)):
            [ZZZ, ooo] = Build_Y(X, index, ZZ[j], "Split")
            for k in range(len(ZZZ)):
                Y.append([Z[i], ZZ[j], ZZZ[k]])
                O.append([o[i], oo[j], ooo[k]])

    return [Y, O]
#------------------------------------------------------------------------------------------
def Test_Model(X, lamda, lm):    #X is the input string, lamda is the parameter
    O_List = ["Nothing","Splitting", "Merging", "Deletion", "Stemming", "Insertion", "Substitution", "Transposition"]
    x = X.split(" ")
    Best_O = []
    Best_y = []

    yi_1 = "<s>"
    index = 0
    poi = ""
    isreplace = 0
    merged_before = 0

    for i in  range(len(x)):
        Best_P = -10000
        Best_oi = ""
        Best_yi = ""

        if merged_before == 1:
            index += 1
            continue

        [Y, O] = Get_Possible_Y(X, i, lm)

        for yi in range (len(Y)):
            zi = Y[yi][len(Y[yi])-1]
            if abs(zi.find('#') - zi.find('-')) == 1:
                continue

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
            score = lamda * f(yi_1, tmp_string, lm) + lamda + lamda *Get_Probability(tmp_string, lm)#* h(yi, oi, X, index)

            tmp_varialbe = Y[yi][len(Y[yi])-1]
            if tmp_varialbe.find('-') != -1:
                fp = zi[:tmp_varialbe.find('-')]    # from beginning to n (n not included)
                sp = zi[tmp_varialbe.find('-')+1:]    # n+1 through end of string
                if sp.find('#') != -1:
                    first = sp[:sp.find('#')]
                    sec = sp[sp.find('#')+1,:]
                    if Get_Probability(first + sec, lm) < -49:
                        continue
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

    #Get_Possible_Y("ho w to mace catt", 1, Language_Model)
    #landa = optimize.minimize(F_Function, [1], method='L-BFGS-B', options=dict({'maxiter':10}))

    #O_landa = landa.x[0]
    O_landa = 1
    x = Test_Model("ho w to mace catt", O_landa, Language_Model)
    print(x)
    print("--------------finish-----------------")