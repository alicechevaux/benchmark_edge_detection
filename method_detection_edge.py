import numpy as np
from sklearn.covariance import GraphicalLasso,GraphicalLassoCV
from scipy.stats import multivariate_normal
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import statsmodels.stats.multitest as smm
from sklearn.covariance import LedoitWolf

def cov2cor(M):
    std_devs = np.sqrt(np.diag(M))

    cor_matrix = M / np.outer(std_devs, std_devs)
    return cor_matrix

def glasso_cv_results(S,Y,results="measure_only",Stype="precision"):
    model = GraphicalLassoCV(cv=5)  # Validation croisée à 5 plis par défaut
    model.fit(Y)
    if Stype=="precision":
        g=model.precision_
    else:
        g=model.covariance_
    n=S.shape[0]
    FP=np.sum((S==0) & (g!=0))/(n*(n-1))
    FN=np.sum((S!=0) & (g==0))/(n*(n-1))
    TP=np.sum((S!=0) & (g!=0))/(n*(n-1))-1/(n-1)
    TN=np.sum((S==0) & (g==0))/(n*(n-1))
    if results=="measure_only":
        return {"Stype":Stype,"FP":FP,"FN":FN,"TN":TN,"TP":TP,"average":TN+TP,"lambda":model.alpha_}
    else:
        return {"Stype":Stype,"g":g,"FP":FP,"FN":FN,"TN":TN,"TP":TP,"average":TN+TP,"lambda":model.alpha_}



def glasso(Y,parameter):
    alpha = parameter  # Coefficient de régularisation (contrôle la sparsité)
    model = GraphicalLasso(alpha=alpha, max_iter=100, tol=1e-4)
    model.fit(Y)

    # Résultats
    precision_matrix = model.precision_  # Matrice de précision (inverse de la covariance)
    return precision_matrix!=0



def glasso_list_results(S,Y,list_parameters,results="measure_only"):
    g_list = [glasso(Y, alpha) for alpha in list_parameters]
    FP=np.zeros(len(g_list))
    FN=np.zeros(len(g_list))
    TP=np.zeros(len(g_list))
    TN=np.zeros(len(g_list))
    for i in range(len(g_list)):
        g=g_list[i]
        n=g.shape[0]
        FP[i]=np.sum((S==0) & (g!=0))/(n*(n-1))
        FN[i]=np.sum((S!=0) & (g==0))/(n*(n-1))
        TP[i]=np.sum((S!=0) & (g!=0))/(n*(n-1))-1/(n-1)
        TN[i]=np.sum((S==0) & (g==0))/(n*(n-1))
    if results=="measure_only":
        return {"plist":list_parameters,"FP":FP,"FN":FN,"TN":TN,"TP":TP,"average":TN+TP}
    else:
        return {"plist":list_parameters,"g_list":g_list,"FP":FP,"FN":FN,"TN":TN,"TP":TP,"average":TN+TP}

def hardthreshold(Y,parameter,ledoitwolf=False):
    if ledoitwolf==True:
        Slw=LedoitWolf().fit(Y)
        cor_mat=cov2cor(Slw.covariance_)
    else:
        cor_mat=cov2cor(np.cov(Y.T))
    thr=parameter

    return cor_mat>thr



def hardthreshold_list_results(S,Y,list_parameters,results="measure_only",ledoitwolf=False):
    g_list = [hardthreshold(Y, thr,ledoitwolf=ledoitwolf) for thr in list_parameters]
    FP=np.zeros(len(g_list))
    FN=np.zeros(len(g_list))
    TP=np.zeros(len(g_list))
    TN=np.zeros(len(g_list))
    for i in range(len(g_list)):
        g=g_list[i]
        n=g.shape[0]
        FP[i]=np.sum((S==0) & (g!=0))/(n*(n-1))
        FN[i]=np.sum((S!=0) & (g==0))/(n*(n-1))
        TP[i]=np.sum((S!=0) & (g!=0))/(n*(n-1))-1/(n-1)
        TN[i]=np.sum((S==0) & (g==0))/(n*(n-1))
    if results=="measure_only":
        return {"plist":list_parameters,"FP":FP,"FN":FN,"TN":TN,"TP":TP,"average":TN+TP}
    else:
        return {"plist":list_parameters,"g_list":g_list,"FP":FP,"FN":FN,"TN":TN,"TP":TP,"average":TN+TP}


def multiplecortest(Y,alpha=0.05,method="BH"):
    p_values=[]
    n_variables=Y.shape[1]
    pairs = []
    rej_matrix = np.ones((n_variables, n_variables), dtype=bool)
    for i in range(n_variables):
        for j in range(i + 1, n_variables):
            # Test de Pearson pour chaque paire (i, j)
            _, p_value = pearsonr(Y[:, i], Y[:, j])
            p_values.append(p_value)
            pairs.append((i, j))

    num_tests = len(p_values)
    if method=="BH":
        rej, p_corr = smm.fdrcorrection(p_values, alpha=alpha, method='indep', is_sorted=False)
    if method=="bonferonni":
        p_bonferroni = np.minimum(np.array(p_values) * num_tests, 1)
        rej=p_bonferroni<alpha
    
    for (i, j), p_val_corr in zip(pairs, rej):
        rej_matrix[i, j] = p_val_corr
        rej_matrix[j, i] = p_val_corr

    return rej_matrix



def multiplecortest_list_results(S,Y,alpha=0.05,list_parameters=["BH","bonferonni"],results="measure_only"):
    g_list = [multiplecortest(Y,alpha, method) for method in list_parameters]
    FP=np.zeros(len(g_list))
    FN=np.zeros(len(g_list))
    TP=np.zeros(len(g_list))
    TN=np.zeros(len(g_list))
    for i in range(len(g_list)):
        g=g_list[i]
        n=g.shape[0]
        FP[i]=np.sum((S==0) & (g!=0))/(n*(n-1))
        FN[i]=np.sum((S!=0) & (g==0))/(n*(n-1))
        TP[i]=np.sum((S!=0) & (g!=0))/(n*(n-1))-1/(n-1)
        TN[i]=np.sum((S==0) & (g==0))/(n*(n-1))
    if results=="measure_only":
        return {"plist":list_parameters,"FP":FP,"FN":FN,"TN":TN,"TP":TP,"average":TN+TP}
    else:
        return {"plist":list_parameters,"g_list":g_list,"FP":FP,"FN":FN,"TN":TN,"TP":TP,"average":TN+TP}


from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages
# import R's utility package
utils = rpackages.importr('utils')
utils.chooseCRANmirror(ind=1)

packnames = ('spcov')

# R vector of strings
from rpy2.robjects.vectors import StrVector

# Selectively install what needs to be install.
# We are fancy, just because we can.
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))


import rpy2.robjects as robjects
robjects.r('''
        # create a function `f`
        f=function(Y,nfolds=5,lambda_values=seq(0,1,by=0.05)){
            n <- nrow(Y)
            folds <- sample(rep(1:nfolds, length.out = n))  # Créer des folds
            cv_results <- sapply(lambda_values, function(lambda) {
                fold_errors <- numeric(nfolds)  # Stocker les erreurs pour chaque fold
                
                for (fold in 1:nfolds) {
                # Séparer les données en train/test
                test_indices <- which(folds == fold)
                train_indices <- setdiff(1:n, test_indices)
                
                Y_train <- Y[train_indices, ]
                Y_test <- Y[test_indices, ]
                
                # Matrice de covariance sur les données d'entraînement
                S_train <- cov(Y_train)
                
                # Estimer la matrice Sigma avec spcov
                fit <- spcov::spcov(diag(diag(S_train)), S_train, lambda = lambda, step.size = 100,backtracking = 0.2)
                Sigma_hat <- fit$Sigma
                
                # Calcul de l'erreur de prédiction sur le set de test
                S_test <- cov(Y_test)
                fold_errors[fold] <- sum((S_test - Sigma_hat)^2)  # Erreur quadratique sur la covariance
                }
                
                mean(fold_errors)  # Moyenne des erreurs sur tous les folds
            })
            
            # Sélection du meilleur lambda
            best_lambda <- lambda_values[which.min(cv_results)]
            S=cov(Y)
            return(spcov::spcov(diag(diag(S)),S,best_lambda,step.size = 100)$Sigma)
        }
        ''')
r_spcov = robjects.r['f']

from rpy2.robjects import numpy2ri

# Activer la conversion automatique entre numpy et R
numpy2ri.activate()

def spcov_cv_results(S,Y,results="measure_only",Stype="precision",nfolds=5):
    g=r_spcov(Y.T,nfolds)
    n=S.shape[0]
    FP=np.sum((S==0) & (g!=0))/(n*(n-1))
    FN=np.sum((S!=0) & (g==0))/(n*(n-1))
    TP=np.sum((S!=0) & (g!=0))/(n*(n-1))-1/(n-1)
    TN=np.sum((S==0) & (g==0))/(n*(n-1))
    if results=="measure_only":
        return {"Stype":Stype,"FP":FP,"FN":FN,"TN":TN,"TP":TP,"average":TN+TP}
    else:
        return {"Stype":Stype,"g":g,"FP":FP,"FN":FN,"TN":TN,"TP":TP,"average":TN+TP}



