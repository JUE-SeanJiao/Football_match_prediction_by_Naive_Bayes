#!/usr/unin/env python

import os
# import sqlite3
# import pandas as pd
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from dateutil.parser import parse
from datetime import datetime

reload(sys)
sys.setdefaultencoding('utf8')
# os.chdir("/Users/ybei/Desktop/data to model project")

### ------------------------- Functions -------------------------
def get_strength(score,opt):
    if opt=='ucl':
        if score<35:
            return 14
        elif score>=95:
            return 1
        else:
            for i in range(2,14):
                if score>=(100-i*5):
                    return i
    elif opt=='dom':
        if score<25:
            return 14
        elif score>=85:
            return 1
        else:
            for i in range(2,14):
                if score>=(90-i*5):
                    return i

def get_matResult(mat,G):
    goalDiff = int(mat[4])-int(mat[5])
    # G = (L-1)/2
    if goalDiff>G:
        return G
    elif goalDiff<-G:
        return -G
    else:
        return goalDiff

def get_WLD(prdd,G):
    prdd = np.flipud(prdd)
    ress = np.argmax(prdd)
    ress = ress-G;
    if ress<0:
        return 0
    elif ress==0:
        return 1
    else:
        return 2

def get_preMatches(mat,tm):
    prevM = []
    for m2 in dom_matches:
        if m2[0]==mat[0] and m2[1]<mat[1]:
            if m2[2]==tm or m2[3]==tm:
                prevM.append(m2)
    for m2 in ucl_matches:
        if m2[0]==mat[0] and m2[1]<mat[1]:
            if m2[2]==tm or m2[3]==tm:
                prevM.append(m2)
    return prevM

# def get_preNPerf(prevM,tm,N):
#     # N = 3
#     dates = []
#     for m2 in prevM:
#         dates.append(m2[1])
#     dates.sort(reverse=True)
#     # print dates
#     # print 'Previous 3:'
#     prevNM = []
#     for m2 in prevM:
#         for j in range(N):
#             if m2[1]==dates[j]:
#                 prevNM.append(m2)
#                 # print m2
#     preRes = np.zeros(N)
#     for mIdx in range(N):
#         m2 = prevNM[mIdx]  
#         if m2[4]>m2[5]:
#             preRes[mIdx] = -1
#         elif m2[4]==m2[5]:
#             preRes[mIdx] = 0
#         else:
#             preRes[mIdx] = 1
#         if m2[3]==tm:
#             preRes[mIdx] = -preRes[mIdx]
#     preRes = preRes+1
#     prevRes = 0
#     for mIdx in range(N):
#         prevRes += 3**mIdx*preRes[mIdx]
#     return int(prevRes)

def get_preNPerf(prevM,tm,N):
    # N = 3
    dates = []
    for m2 in prevM:
        dates.append(m2[1])
    dates.sort(reverse=True)
    # print dates
    # print 'Previous 3:'
    prevNM = []
    for m2 in prevM:
        for j in range(N):
            if m2[1]==dates[j]:
                prevNM.append(m2)
                # print m2
    scs = 0
    for mIdx in range(N):
        m2 = prevNM[mIdx]
        if m2[4]==m2[5]:
            scs += 1
        if m2[2]==tm:
            if m2[4]>m2[5]:
                scs += 3
        else:
            if m2[4]<m2[5]:
                scs += 3
    # print scs
    if N==3:
        if scs>=7:
            return 0
        elif scs>=5:
            return 1
        elif scs>=3:
            return 2
        else:
            return 3

### ------------------------------------------------------------

teams = {}  ### Name-ID ###
ucl_matches = []  ### [season,date,home_team_id,away_team_id,home_team_goal,away_team_goal]


asgn = 20001
with open('UCL-matches.txt','r') as f:
	for line in f:
		if line[0]=='2' or len(line)==1:
			continue
		mat = line.rstrip('\n').split(',')
		hTm = mat[0].split('-')
		if hTm[0] not in teams:
			if hTm[1]!='':
				teams[hTm[0]] = hTm[1]
			else:
				teams[hTm[0]] = str(asgn)
				asgn += 1
		aTm = mat[1].split('-')
		if aTm[0] not in teams:
			if aTm[1]!='':
				teams[aTm[0]] = aTm[1]
			else:
				teams[aTm[0]] = str(asgn)
				asgn += 1
	f.seek(0)
	for line in f:
		if line[0]=='2' or len(line)==1:
			if line[0]=='2':
				tmp = line.split(':')
				season = tmp[0]
			continue
		mat = line.rstrip('\n').split(',')
		hTm = mat[0].split('-')
		aTm = mat[1].split('-')
		sc = mat[3].split(':')
		ucl_matches.append([season,parse(mat[4]),teams[hTm[0]],teams[aTm[0]],sc[0],sc[1]])


country2id = {}
with open('Countries.csv','r') as f:
	f.readline()
	for line in f:
		c = line.rstrip('\n').split(',')
		if c[1] not in country2id:
			country2id[c[1]] = c[2]
dom_matches = []  ### [country,season,date,home_team_id,away_team_id,home_team_goal,away_team_goal]
with open('Match.csv','r') as f:
	f.readline()
	for line in f:
		mat = line.rstrip('\n').split(',')
		dt = (mat[4].split(' '))[0]
		dom_matches.append([mat[3],parse(dt),mat[5],mat[6],mat[7],mat[8],country2id[mat[2]]])



season = {'2008':0,'2009':1,'2010':2,'2011':3,'2012':4,'2013':5,'2014':6,'2015':7}
league_name = ['England','France','Germany','Portugal','Spain','Italy','Netherlands']
seasons = ['2011/2012','2012/2013','2013/2014','2014/2015','2015/2016']
seasons2 = ['2008/2009','2009/2010','2010/2011','2011/2012','2012/2013','2013/2014','2014/2015','2015/2016']
seasons3 = ['2011-2012','2012-2013','2013-2014','2014-2015','2015-2016']

point_table = {}
team2country = {}

for cty in league_name:
    with open(cty+'.txt','r') as f:
        for line in f:
            if line[0]=='2' or len(line)==1:
                continue
            c = line.rstrip('\n').split(',')
            d = c[0].split('-')
            if d[1] not in point_table.keys():
                point_table[d[1]] = np.zeros(8)
                team2country[d[1]] = cty
        f.seek(0)
        for line in f:
            if line[0]=='2' or len(line)==1:
                if line[0]=='2':
                    tmp = line.split(',')
                season_now = season[tmp[0]]
                match = float(tmp[1])
                continue
       	    c = line.rstrip('\n').split(',')
            d = c[0].split('-')
            point_table[d[1]][season_now] = float(c[1])/match

with open('other-leagues.txt','r') as f:
    for line in f:
        if line[0]=='2' or len(line)==1:
            continue
        c = line.rstrip('\n').split(',')
        if teams[c[0]] not in point_table.keys():
            point_table[teams[c[0]]] = np.zeros(8)
        if teams[c[0]] not in team2country:
            team2country[teams[c[0]]] = c[1]
    f.seek(0)
    for line in f:
        if line[0]=='2' or len(line)==1:
            if line[0]=='2':
                tmp = line.split(',')
            season_now = season[tmp[0]]
            continue
        c = line.rstrip('\n').split(',')
        for i in range(2,6):
            e = c[i].split(':')
            point_table[teams[c[0]]][season_now-5+i] = float(e[0])/float(e[1])


league_coeff={}
with open('league-coefficients.txt','r') as f:
    f.readline()
    for line in f:
        c = line.rstrip('\n').split(',')
        league_coeff[c[0]] = np.zeros(8)
        d = c[1]
        e = d.split(':')
        for i in range(8):
            league_coeff[c[0]][i] = e[i]

bias_strg = {}
for i,pt in point_table.items():
    tmp = league_coeff[team2country[i]]
    tmp2 = np.zeros(tmp.shape)
    for j in range(len(tmp)):
        tmp2[j] = tmp[j]**(3/float(4))
    bias_strg[i] = 5*pt*tmp2

# wgt = np.array([1/3,1/2,1,1])
wgt = np.array([1/3,1/2,1,0.6])
wgt = np.exp(wgt)
wgt = np.transpose(wgt/np.sum(wgt))

wgt2 = np.array([1/3,1/2,1])
wgt2 = np.exp(wgt2)
wgt2 = np.transpose(wgt2/np.sum(wgt2))


# print '=================================================='
# print 'UCL Team-id:',len(teams.keys())
# for t in teams:
#   print t,'|',teams[t]

# print '=================================================='
# print 'UCL matches:',len(ucl_matches)
# for m in ucl_matches:
#   print m

# print '=================================================='
# print 'Domestic matches:',len(dom_matches)
# for m in dom_matches:
#   print m

# print '=================================================='
# print 'Team-league affliation:'
# for i,cty in team2country.items():
#     print i,'|',cty

# print '=================================================='
# print 'Domestic point table (per match)'
# for i,pt in point_table.items():
#     print 'team',i,'\n',pt

# print '=================================================='
# print 'Country coefficients:'
# for cty,pt in league_coeff.items():
#     print cty,pt

# print '=================================================='
# print 'Team bias_strg in season 2015/2016:'
# for i,st in bias_strg.items():
#     print 'team',i,'\n',st[7]  

# # output = []
# for i in range(len(seasons)):
#     print '=================================================='
#     print '* Team strength in season',seasons[i]
#     teamTmp = {}
#     for m in ucl_matches:
#         tm = m[2]
#         if m[0]==seasons[i] and tm not in teamTmp:
#             teamTmp[tm] = np.dot(bias_strg[tm][i:i+3],wgt2)
#     for j,sc in teamTmp.items():
#         print 'team',j,':',get_strength(sc,'ucl')
#     # output.append(teamTmp)
#     scores = teamTmp.values()
#     scores.sort()

#     fig = plt.figure()
#     fig,ax = plt.subplots()
#     ax.plot(range(1,33),scores,'.')
#     plt.xlabel('team')
#     plt.ylabel('scores')
#     intervals = float(5)
#     loc = plticker.MultipleLocator(base=intervals)
#     # ax.xaxis.set_major_locator(loc)
#     ax.yaxis.set_major_locator(loc)
#     ax.grid(which='major', axis='both', linestyle='-')
#     plt.savefig(seasons3[i]+'.png')
#     plt.close(fig)
# # np.save('ucl-scores.npy',output)





print '=================================================='

##### NAIVE #####

N = 3 ## previous N matches
L = 9 ## L-level match result
G = (L-1)/2
trnUCL = range(len(seasons)) ## trn and tes ucl seasons
# tesUCL = [0] ## 0,1,2,3,4
tesUCL = range(len(seasons))
for ss in tesUCL:
    trnUCL.remove(ss)
RR = 1 ## treat a ucl-match as RR training samples


# count_Nai = np.zeros([L,14,14,3**N,3**N])
count_Nai = np.zeros([L,14,14,N+1,N+1])
count1 = np.zeros([L,14,14])
count2 = np.zeros([L,N+1,N+1])
for i in range(len(seasons)):
    # print '*',seasons[i]
    for m in dom_matches:
        if m[0]==seasons[i]:
            # print '$ match:',m
            htm = m[2]
            atm = m[3]
            sc = np.dot(point_table[htm][i:i+4],wgt)*38
            # sc = np.dot(point_table[htm][i:i+3],wgt2)*38
            h_strg = get_strength(sc,'dom')
            sc = np.dot(point_table[atm][i:i+4],wgt)*38
            # sc = np.dot(point_table[atm][i:i+3],wgt2)*38
            a_strg = get_strength(sc,'dom')
            res = get_matResult(m,G)
            count1[res+G,h_strg-1,a_strg-1] += 1

            prevM_H = get_preMatches(m,htm)
            prevM_A = get_preMatches(m,atm)
            if len(prevM_H)<N or len(prevM_A)<N:
                # print 'Home',htm+':',h_strg,'| Away',atm+':',a_strg,'| result:',res
                continue
            prevN_H = get_preNPerf(prevM_H,htm,N)
            prevN_A = get_preNPerf(prevM_A,atm,N)

            # print 'Home',htm+':',h_strg,'( prev-N',prevN_H,') | Away',atm+':',a_strg,'( prev-N',prevN_A,') | result:',res
            count_Nai[res+G,h_strg-1,a_strg-1,prevN_H,prevN_A] += 1
            count2[res+G,prevN_H,prevN_A] += 1
    #         break
    #     break
    # break


# for i in trnUCL:
#     # print '*',seasons[i]
#     for m in ucl_matches:
#         if m[0]==seasons[i]:
#             # print '$ match:\n',m
#             htm = m[2]
#             atm = m[3]
#             sc = np.dot(bias_strg[htm][i:i+3],wgt2)
#             h_strg = get_strength(sc,'ucl')
#             sc = np.dot(bias_strg[atm][i:i+3],wgt2)
#             a_strg = get_strength(sc,'ucl')
#             res = get_matResult(m,G)
#             count1[res+G,h_strg-1,a_strg-1] += RR

#             if (team2country[htm] not in league_name) or (team2country[atm] not in league_name):
#                 continue
#             prevM_H = get_preMatches(m,htm)
#             prevM_A = get_preMatches(m,atm)
#             if len(prevM_H)<N or len(prevM_A)<N:
#                 continue
#             prevN_H = get_preNPerf(prevM_H,htm,N)
#             prevN_A = get_preNPerf(prevM_A,atm,N)

#             # print 'Home',htm+':',h_strg,'( prev-3',prevN_H,') | Away',atm+':',a_strg,'( prev-3',prevN_A,') | result:',res
#             count_Nai[res+G,h_strg-1,a_strg-1,prevN_H,prevN_A] += RR
#             count2[res+G,prevN_H,prevN_A] += RR




# print '=================================================='
count_Nai += 1
count1 += 1
count2 += 1
count_Nai[4,:] += 3
count1[4,:] += 3
count2[4,:] += 3
Nai_cpd = np.zeros(count_Nai.shape)
Nai1_cpd = np.zeros(count1.shape)
for i1 in range(14):
    for i2 in range(14):
        sumTmp1 = np.sum(count1[:,i1,i2])
        if sumTmp1!=0:
            Nai1_cpd[:,i1,i2] = count1[:,i1,i2]/sumTmp1
        # print i1,i2,':',Nai1_cpd[:,i1,i2]
        for i3 in range(N+1):
            for i4 in range(N+1):
                sumTmp = np.sum(count_Nai[:,i1,i2,i3,i4])
                if sumTmp!=0:
                    Nai_cpd[:,i1,i2,i3,i4] = count_Nai[:,i1,i2,i3,i4]/sumTmp
                # print i1,i2,i3,i4,':',Nai_cpd[:,i1,i2,i3,i4]
Nai2_cpd = np.zeros(count2.shape)
for i3 in range(N+1):
    for i4 in range(N+1):
        sumTmp2 = np.sum(count2[:,i3,i4])
        if sumTmp2!=0:
            Nai2_cpd[:,i3,i4] = count2[:,i3,i4]/sumTmp2
        # print i3,i4,':',Nai2_cpd[:,i3,i4]



print '=================================================='
print 'Predict:'
correct = 0
total = 0
correct1 = 0
total1 = 0
correct2 = 0
total2 = 0
for m in ucl_matches:
    for i in tesUCL:
        if m[0]==seasons[i]:
            print '------------------------- Match -------------------------\n',m
            htm = m[2]
            atm = m[3]
            gDiff = int(m[4])-int(m[5])
            total1 += 1
            sc = np.dot(bias_strg[htm][i:i+3],wgt2)
            h_strg = get_strength(sc,'ucl')
            sc = np.dot(bias_strg[atm][i:i+3],wgt2)
            a_strg = get_strength(sc,'ucl')
            # res = get_matResult(m,G)
            prd1 = Nai1_cpd[:,h_strg-1,a_strg-1]
            if np.sum(prd1)!=0:
                res_prd1 = np.array([np.sum(prd1[G+1:]),prd1[G],np.sum(prd1[:G])])
            else:
                res_prd1 = np.array([0.35,0.3,0.35])
            # res1 = get_WLD(prd1,G)
            res1 = np.argmax(res_prd1)
            print 'Strength estimate:',res1
            print 'H:',h_strg,'| A:',a_strg
            print 'H-win:',res_prd1[0],'| H-draw:',res_prd1[1],'| H-lose:',res_prd1[2]
            # print res+4
           
            if (gDiff>0 and res1==0) or (gDiff==0 and res1==1) or (gDiff<0 and res1==2):
                correct1 += 1
                print '$$ correct'

            if (team2country[htm] not in league_name) or (team2country[atm] not in league_name):
                continue
            prevM_H = get_preMatches(m,htm)
            prevM_A = get_preMatches(m,atm)
            if len(prevM_H)<N or len(prevM_A)<N:
                # break
                continue
            prevN_H = get_preNPerf(prevM_H,htm,N)
            prevN_A = get_preNPerf(prevM_A,atm,N)

            total += 1
            total2 += 1
            prd2 = Nai2_cpd[:,prevN_H,prevN_A]
            if np.sum(prd2)!=0:
                res_prd2 = np.array([np.sum(prd2[G+1:]),prd2[G],np.sum(prd2[:G])])
            else:
                res_prd2 = np.array([0.35,0.3,0.35])
            # res2 = get_WLD(prd2,G)
            res2 = np.argmax(res_prd2) 
            # print 'Previous N-matches estimate:',res2
            # print 'H:',prevN_H,'| A:',prevN_A
            # print 'H-win:',res_prd2[0],'| H-draw:',res_prd2[1],'| H-lose:',res_prd2[2]
          
            if (gDiff>0 and res2==0) or (gDiff==0 and res2==1) or (gDiff<0 and res2==2):
                correct2 += 1
                # print '$$ correct'

            prd = Nai_cpd[:,h_strg-1,a_strg-1,prevN_H,prevN_A]
            if np.sum(prd)!=0:
                res_prd = np.array([np.sum(prd[G+1:]),prd[G],np.sum(prd[:G])])
            else:
                res_prd = np.array([0.35,0.3,0.35])
            # www = 0.9
            # res_prd = res_prd1*www+res_prd2*(1-www)
            # res_prd = res_prd1*res_prd2
            # res_prd = res_prd/np.sum(res_prd)
            # res = get_WLD(prd,G)
            res = np.argmax(res_prd)
            # print 'Strg+PrevN estimate:',res
            # print 'H:',h_strg,'| A:',a_strg
            # print 'H:',prevN_H,'| A:',prevN_A
            # print 'H-win:',res_prd[0],'| H-draw:',res_prd[1],'| H-lose:',res_prd[2]
            
            if (gDiff>0 and res==0) or (gDiff==0 and res==1) or (gDiff<0 and res==2):
                correct += 1
                print '$$ correct'

            # break

print 'Accuracy1:',correct1/float(total1)
# print 'Accuracy2:',correct2/float(total2)
# print 'Accuracy:',correct/float(total)












