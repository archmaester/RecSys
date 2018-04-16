#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division 
import cPickle as pickle 
import numpy as np
import pandas as pd
from enum import Enum
from ast import literal_eval
from sklearn.decomposition import NMF
import sklearn.metrics.pairwise as skm
from sklearn.cluster import KMeans, SpectralClustering
import math 
import scipy
from scipy.sparse import csr_matrix
import kmeans
from collections import Counter


class Model(object):
    def __init__(self, G, R, C, C_k, W, U, V, Users):
        self.G = G
        self.R = R
        self.C = C
        self.C_k = C_k
        self.W = W
        self.U = U
        self.V = V
        self.Users = Users

class Graph(object):
    def __init__(self, name = None):
        # Mainitaining AdjList of nodes hashed on Node object
        self.name = name

    def load_userdb_csv(self,fname, max_mov):

        frame = pd.read_csv(fname)

        users = frame['userId']
        movies = frame['movieId']        
        self.movId2vir = {}
        self.vir2movId = {}
        # Next movie for each user
        self.test_set = {}
        usr2Id = {}
        vir_m_id = 0
        tmpM = []
        tmpU = [] 

        tempU = 1 
        curr_us = -1
        ucnt = -1
        for ii in range(len(users)):           
            if curr_us != users[ii]:
                ucnt += 1
                usr2Id[users[ii]] = ucnt
                tempU = 1
                curr_us = users[ii]

            m = movies[ii]
            if m not in self.movId2vir and tempU < 4 * max_mov:
                self.movId2vir[m] = vir_m_id
                self.vir2movId[vir_m_id] = m
                vir_m_id += 1
            if tempU < max_mov:
                # idx = (users[ii] - 1)
                idx = ucnt
                if idx not in self.test_set:
                    self.test_set[idx] = []
                self.test_set[idx].append(self.movId2vir[m])
            # else:
            elif tempU < 4 * max_mov:
                tmpM.append(self.movId2vir[m])
                # tmpU.append(users[ii])
                tmpU.append(ucnt)                   
            
            tempU += 1

        ones = np.ones(len(tmpU))
        users = np.array(tmpU)
        print("Total ", ucnt)

        self.R = np.array(csr_matrix((ones, (users, tmpM)), shape=(ucnt + 1, len(self.movId2vir))).todense())
        print("User done.", len(self.movId2vir), len(self.test_set[0]), self.R.shape)

    def load_act_csv(self,fname):
        frame = pd.read_csv(fname)

        movies = frame['Id']
        actors = frame['Actors']
        actId2vir = {}
        vir_a = 0
        tmpMA = []
        tmpA = [] 

        for ii in range(len(movies)):

            movie = movies[ii]
            if movie in self.movId2vir:

                act_list = literal_eval(actors[ii])

                for a in act_list:
                    if a not in actId2vir:
                        actId2vir[a] = vir_a
                        vir_a += 1
                    tmpMA.append(self.movId2vir[movie])
                    tmpA.append(actId2vir[a])

        ones = np.ones(len(tmpMA))
        self.MA = csr_matrix((ones, (tmpMA, tmpA)), shape=(len(self.movId2vir), len(actId2vir)))

    def load_dir_csv(self,fname):
        frame = pd.read_csv(fname)
        movies = frame['Id']

        directors = frame['Directors']
        dirId2vir = {}
        vir_d = 0
        tmpMD = []
        tmpD = []

        for ii in range(len(movies)):
            movie = movies[ii]
            if movie in self.movId2vir:
                dir_list = literal_eval(directors[ii])
    
                for d in dir_list:
                    if d not in dirId2vir:
                        dirId2vir[d] = vir_d
                        vir_d += 1
                    tmpMD.append(self.movId2vir[movie])
                    tmpD.append(dirId2vir[d])

        ones = np.ones(len(tmpMD))
        self.MD = csr_matrix((ones, (tmpMD, tmpD)), shape=(len(self.movId2vir), len(dirId2vir)))

    def load_genre_csv(self, fname):
        frame = pd.read_csv(fname)
        movies = frame['Id']
        genres = frame['Genres']
        gnrId2vir = {}
        vir_g = 0
        tmpMG = []
        tmpG = []
 
        for ii in range(len(movies)):
            movie = movies[ii]
            if movie in self.movId2vir:
                gnr_list = literal_eval(genres[ii])

                for g in gnr_list:
                    if g not in gnrId2vir:
                        gnrId2vir[g] = vir_g
                        vir_g += 1
                    tmpMG.append(self.movId2vir[movie])
                    tmpG.append(gnrId2vir[g])

        ones = np.ones(len(tmpMG))
        self.MG = csr_matrix((ones, (tmpMG, tmpG)), shape=(len(self.movId2vir), len(gnrId2vir)))

        
    def load_moviedb_csv(self, fname):
        # Load using pandas
        frame = pd.read_csv(fname)

        movies = frame['Id']
        actors = frame['Actors']
        directors = frame['Directors']
        genres = frame['Genres']


        actId2vir = {}
        dirId2vir = {}
        gnrId2vir = {}

        vir_a = 0
        vir_d = 0
        vir_g = 0

        tmpMA = []
        tmpMD = []
        tmpMG = []
        tmpA = []
        tmpD = []
        tmpG = []
        for ii in range(len(movies)):

            movie = movies[ii]
            if movie in self.movId2vir:

                act_list = literal_eval(actors[ii])
                dir_list = literal_eval(directors[ii])
                gnr_list = literal_eval(genres[ii])

                for a in act_list:
                    if a not in actId2vir:
                        actId2vir[a] = vir_a
                        vir_a += 1
                    tmpMA.append(self.movId2vir[movie])
                    tmpA.append(actId2vir[a])

                for d in dir_list:
                    if d not in dirId2vir:
                        dirId2vir[d] = vir_d
                        vir_d += 1
                    tmpMD.append(self.movId2vir[movie])
                    tmpD.append(dirId2vir[d])

                for g in gnr_list:
                    if g not in gnrId2vir:
                        gnrId2vir[g] = vir_g
                        vir_g += 1
                    tmpMG.append(self.movId2vir[movie])
                    tmpG.append(gnrId2vir[g])

        ones = np.ones(len(tmpMA))
        self.MA = csr_matrix((ones, (tmpMA, tmpA)), shape=(len(self.movId2vir), len(actId2vir)))
        
        ones = np.ones(len(tmpMD))
        self.MD = csr_matrix((ones, (tmpMD, tmpD)), shape=(len(self.movId2vir), len(dirId2vir)))
        
        ones = np.ones(len(tmpMG))
        self.MG = csr_matrix((ones, (tmpMG, tmpG)), shape=(len(self.movId2vir), len(gnrId2vir)))

        print("Done for Moviedb ")

    def get_diffusion_matrix(self, metapath):
        n, m = self.R.shape

        path_cnt = scipy.sparse.identity(m, format = 'csr')

        # metapath is list ['MAM', 'MDM']
        for entry in metapath:
            mat1 = None
            if entry == 'MAM':
                mat1 = self.MA.dot(self.MA.transpose())

            elif entry == 'MDM':
                mat1 = self.MD.dot(self.MD.transpose())

            elif entry == 'MGM':
                mat1 = self.MG.dot(self.MG.transpose())

            elif entry == 'MUM':
                mat1 = csr_matrix((self.R.T).dot(self.R))

            path_cnt = path_cnt.dot(mat1)
        path_cnt = np.array(path_cnt.todense())
        print(path_cnt.shape)

        R_t = np.zeros((n,m))
        tmp = np.zeros((m,m))
        
        for jj in range(m):
            # print(jj)
            for kk in range(m):

                t1 = path_cnt[kk, jj]
                t2 = path_cnt[kk, kk]
                t3 = path_cnt[jj, jj]

                if t2 + t3 == 0:
                    continue

                tmp[kk, jj] += (2 * t1) / (t2 + t3)
                
        R_t = np.matmul(self.R, tmp)


        return R_t 
# we get U,V for each metapath 
#item_dict is vir2movId 
# d is number of dimensions

def get_feature(G, metapaths, d):

    L = len(metapaths)
    m, n = G.R.shape  # m: users n:movies
    U = np.zeros((L, m, d))
    V = np.zeros((L, d, n))
    for ii in range(L):
        print("Diffusing metapath", ii)
        R_t = G.get_diffusion_matrix(metapaths[ii])
        print("Successfully calculated the Diffusion Matrix")
        model = NMF(n_components = d,init='random')
        print("Successfully decomposed the Diffusion Matrix")
        print(R_t.shape)
        U[ii] = model.fit_transform(R_t)
        V[ii] = model.components_
        print("Done All Diffusion", ii)
    print("Done All Diffusion")
    return (U,V)

def get_user_clusters3(R, k, d = 100):
    model = NMF(n_components = d,init='random')
    print("Successfully decomposed the Implicit Feedback Matrix")
    U = model.fit_transform(R)
    labels, C_k = kmeans.kmeans(U, k)
    cnt = Counter(labels)
    print(cnt)
    print("K-Means Clustering Done")
    return (U, labels, C_k )


def get_user_clusters2(R, k, d = 100):
    model = NMF(n_components = d,init='random')
    print("Successfully decomposed the Implicit Feedback Matrix")
    U = model.fit_transform(R)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(U)
    print("K-Means Clustering Done")
    return (U, kmeans.labels_, kmeans.cluster_centers_ )


def get_user_clusters(R, k, d = 100):
    model = NMF(n_components = d,init='random')
    print("Successfully decomposed the Implicit Feedback Matrix")
    #R = np.matmul(R, R.T)
    #R[R>1] = 1
    #print(len(R[R>1]))
    U = model.fit_transform(R)
    sc = SpectralClustering(n_clusters=k, gamma = 0.05, affinity = 'rbf').fit(U)
    print("Spectral Clustering Done")
    C_k = np.zeros((k, d))
    for i in range(k):
        print(len(U[sc.labels_ == i]))
        C_k[i] = np.mean(U[sc.labels_ == i], axis = 0)
        
    return (U, sc.labels_, C_k )

def grad_get_r_iab(w, U,V,i,a,b, L):
    grad_r = np.zeros(L)
    for q in range(L):
        ra = np.matmul(U[q,i,:], V[q,:,a])
        rb = np.matmul(U[q,i,:], V[q,:,b])
        grad_r[q] = ra - rb

    return ( grad_r)

def get_r_iab(w, U, V, i, a, b, L):
    ra = 0
    rb = 0
    for q in range(L):
        ra += w[q] * np.matmul(U[q,i,:], V[q,:,a])

    for q in range(L):
        rb += w[q] * np.matmul(U[q,i,:], V[q,:,b])

    return (ra-rb)

def sigmoid(x):
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    else:
        return (math.exp(x) / (1 + math.exp(x)))


def get_R_iab(w, U,V, L):
    _, n, d = U.shape
    _, d, m = V.shape
    ra = np.zeros((n, m))
    for q in range(L):
        tmp = np.matmul(U[q,:,:], V[q,:,:])
        ra += w[q] * tmp

    return ra

def grad_get_R_iab(w, U,V,L):
    grad_r = np.zeros(L)
    for q in range(L):
        ra = np.matmul(U[q,:,:], V[q,:,:])

    return ( grad_r)

def get_obj(w, R, C_k, U, V, L, lambda1):
    
    # print C_k.shape 
    # print U.shape 
    n_m = 100
    n_u = min(50, len(C_k))
    CC_k = C_k[0:n_u]
    U = U[:,CC_k,:]

    # U = U[:,0:32]
    V = V[:,:,0:n_m]
    
    RA = get_R_iab(w, U, V, L)
    t3 = 0
    for u in range(n_u):
        for i in range(n_m):
            for j in range(n_m):
                if R[CC_k[u], i] > R[CC_k[u], j]:
                    t3 += np.log(sigmoid(RA[u,i] - RA[u,j]))
                    
    obj = t3 + lambda1*np.linalg.norm(w)
    
    return obj
    
def train(U,V, R,C_k,lambda1, lr = 0.01, eps = 0.01):
    
    L, m, d = U.shape
    L, _, n = V.shape
    len_ck = len(C_k)
    
    converge = False
    w = np.ones(L) * (1.0 / L)
    grad_norm = 0
    tt = 0
    ind = np.arange(n)
    b_size = min(32, len_ck)

    print(b_size)
    tmpp = np.arange(len_ck)
    obj = 0
    while (not converge) and tt < 5000:
        tt += 1
        grad = np.zeros(L)
        np.random.shuffle(tmpp)
        batch = C_k[tmpp[0:b_size]]

        for kk in batch:
    
    
            ind_t = ind[R[kk] == 1]
            ind_f = ind[R[kk] == 0]
    
            p = np.random.randint(len(ind_t))
            q = np.random.randint(len(ind_f))
    
            a = ind_t[p]
            b = ind_f[q]
    
            t1 = sigmoid(-get_r_iab(w, U, V, kk, a, b, L))
            t2 = grad_get_r_iab(w, U, V, kk, a, b, L) * t1
            grad += t2

        grad = -grad + lambda1 * w

        w = w - (1.0 / tt) * grad
        new_obj = get_obj(w, R, C_k, U, V, L, lambda1)
        if np.abs(new_obj - obj) < eps:
            converge = True
        print(tt,np.abs(new_obj - obj))
        obj = new_obj

    return (w)

def learn_model(U,V,C,R, K, lambda1 = 1, lr = 0.01, eps = 0.01):
    L, m, d = U.shape
    L, _, n = V.shape
    w = (1/L)*np.ones((K, L))
    ind = np.arange(len(C))
    for ii in range(K):
        
        C_k = ind[C==ii]
        print("Training for Cluster {}".format(ii))
        w[ii] = train(U, V, R,C_k,lambda1,lr,eps)

    return w 

def get_r_measure(ui,i, j, K, L, W, U, V, mu_k, C):
    r = 0
    L, n, d = U.shape
    ui = np.reshape(ui, (-1,1))
    ui_norm = np.linalg.norm(ui)
    sim = 0
    for k in range(K):
        t1 = np.reshape(mu_k[k],(-1,1))
        t1_norm = np.linalg.norm(ui)
        sim = np.dot(t1.T, ui) / (ui_norm * t1_norm)
        '''if C[i] == k:
            sim = 1
        else:
            sim = 0'''
        tmp = 0
        for q in range(L):
            tmp += W[k,q] * np.matmul(U[q,i,:], V[q,:,j])
        r += sim*tmp
    return (r)

def get_top_movies(ui, i, K, W, U, V, mu_k, t = 10):
    
    L, _, n = V.shape
    S = np.zeros(n)
    for j in range(n):
        S[j] = get_r_measure(ui, i, j, K, L, W, U, V, mu_k)

    Mov = np.argsort(S)
    print(np.mean((S[0:t])))
    print(Mov[0:t])

def evaluation(model, K , d):
    
    L, _, n = model.V.shape
    nmf = NMF(n_components = d,init='random')
    User = model.Users
    m = len(model.G.test_set)
    t = 10
    mrr = 0
    tcnt = 0
    S = np.zeros(n)
    # Test MRR
    for user in model.G.test_set:
        nmov = model.G.test_set[user]
        for j in range(n):
            if model.R[user][j] == 0:
                S[j] = get_r_measure(User[user], user, j, K, L, model.W, model.U, model.V, model.C_k, model.C)
    
        Mov = np.argsort(-S)
        ind = np.arange(n)
        ind_t = []
        for nm in nmov:
            ind_t.append(ind[Mov == nm])
        ind_t = np.array(ind_t, dtype = 'float64').ravel()
        tcnt = len(ind_t)
        if tcnt > 0:
            ind_t += 1.0
            print(ind_t)
            r_t = np.reciprocal(ind_t)
            mrr += np.sum(r_t)

    print(mrr/m)
    # Training MRR
    tmrr = 0
    S = np.zeros(n)
    indd = np.arange(n)
    for user in range(User.shape[0]):
        nmov = indd[model.R[user] == 1]
        for j in range(n):
            S[j] = get_r_measure(User[user], user, j, K, L, model.W, model.U, model.V, model.C_k, model.C)
        Mov = np.argsort(-S)
        ind = np.arange(n)
        ind_t = []
        for nm in nmov:
            ind_t.append(ind[Mov == nm])
        ind_t = np.array(ind_t, dtype = 'float64').ravel()
        tcnt = len(ind_t)
        if tcnt > 0:
            ind_t += 1.0
            print(ind_t)
            r_t = np.reciprocal(ind_t)
            tmrr += np.sum(r_t)

    print("Evaluation End...", tcnt, (tmrr/m))
    return (mrr / m)
               
def graph_save(G):
    fp = open(G.name, 'wb')
    pickle.dump(G, fp)
    fp.close()

def graph_load(fname):
    fp = open(fname, 'rb')
    G = pickle.load(fp)
    fp.close()
    return G


def pickle_save(object,filename):
    fp = open(filename,'wb')
    pickle.dump(object,fp)
    fp.close()

def pickle_load(filename):
    fp = open(filename,'rb')
    object1 = pickle.load(fp)
    fp.close()
    return object1

def get_adj_matrix(filename,col1,col2):

    frame = pd.read_csv(filename)
    col1 = frame[col1]
    col2 = frame[col2]

    W = np.matrix((len(col1,)))


if __name__ == '__main__':

    num_mov = 10 # No of test ratings per user    
    d = 20  # Low rank factor
    k = 10 # No of clusters

        
    G = Graph("RECSYS") 
#
    G.load_userdb_csv('./User_Movies.csv', num_mov)
    G.load_act_csv('./Movie_Actors.csv')
    G.load_dir_csv('./Movie_Directors.csv')
    G.load_genre_csv('./Movie_Genres.csv') 

    metapaths = [['MDM'], ['MAM'],['MGM'],['MUM']]
#        metapaths = [['MDM','MAM'],['MAM','MGM','MDM'],['MDM'], ['MAM'],['MGM'],['MUM']]

    [U,V] = get_feature(G, metapaths, d)
    print("feature extraction done")
    [Users,C,cluster_centers] = get_user_clusters3(G.R, k, d_r)

    model = Model(G, 0, C, cluster_centers, 0, U, V, Users)
    
    pickle_save(model, "model-5")

    print("training starts")
    #Training Model
    W = learn_model(model.U,model.V,model.C, model.G.R,k, lambda1 = 0.1, lr = 0.01, eps = 0.01)
    print("training over")
    pickle_save(W, "weight-5")
    
    
    model = pickle_load("model-5")
    W = pickle_load("weight-5")
    model.R = model.G.R
    model.W = W
    
    # Evaluating our model
    print(evaluation(model, k, d))

