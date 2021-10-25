import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import random

from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.utils import resample

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek

from imblearn.pipeline import make_pipeline
import statsmodels.api as sm


class Pipeline_observational():
    def __init__(self,X,y,clf,obs_data,random_state=None):
        self.obs_data = obs_data
        self.X = X
        self.y = y
        self.clf = clf
        self.threshold = threshold = 0.8
        self.random_state = random_state
        self.parameter_grid = parameter_grid = [{'kernel': ['poly', 'rbf'],
                                                'C': [0.01, 0.1,1, 10, 100,],
                                                'gamma': [.1, .01, 1e-3]}, ]

        self.symptom_names = ['Covid-Recovered', 'Covid-Positive', 'No-Taste/Smell', 'Fever', 'Headache', 'Pneumonia', 'Stomach', 'Myocarditis', 'Blood-Clots', 'Death']

    def find_alpha(self, beta,p):
        """ Given beta and a mean probability p, compute and return the alpha of a beta distribution. """
        return beta*p/(1-p)

    def find_efficacy(self, group_pos: pd.DataFrame, group_neg: pd.DataFrame, symptom, prior_probs):
        if isinstance(symptom, int):
            symptom_index = symptom
            symptom_name = group_pos.keys()[symptom]
        else:
            symptom_name = symptom
            symptom_index = list(group_pos.keys()).index(symptom)

        group_pos_count = np.sum(group_pos[symptom_name] * group_pos.iloc[:,1])
        group_neg_count = np.sum(group_neg[symptom_name] * group_neg.iloc[:,1])

        v = group_pos_count/len(group_pos)
        n_v = group_neg_count/len(group_neg)

        if n_v == 0:
            print(f'{v=}, {n_v=}: Division by zero')
            return

        IRR = v/n_v

        #print(v, n_v)
        efficacy = 100*(1- IRR)

        N = 100_000
        beta = 1
        p = prior_probs[symptom_index]
        alpha = self.find_alpha(beta,p)

        samples_group_pos = stats.beta.rvs(alpha + group_pos_count, beta + len(group_pos) - group_pos_count, size=N)
        samples_group_neg = stats.beta.rvs(alpha + group_neg_count, beta + len(group_neg) - group_neg_count, size=N)

        samples_ve = 100 * (1 - samples_group_pos/samples_group_neg)
        lower = np.percentile(samples_ve, 2.5)
        upper = np.percentile(samples_ve, 97.5)

        if efficacy >= lower and efficacy <= upper:
            status = 'not rejected'
        else:
            status = 'rejected'

        print(f'{symptom_name:15s}: {efficacy:3.3f} - ({lower:3.3f}, {upper:3.3f}) - {status}')

    def run_efficacy(self, vacced, un_vacced,prior_probs):
        for i, s in enumerate(self.symptom_names):
            self.find_efficacy(vacced,un_vacced,i,prior_probs)


sym_g = np.random.choice(2,size=[100_000,10])
y = np.random.choice(2,size=[100_000,1])

vac_g = np.zeros([100_000,3])
for i,v in enumerate(y):
    if v == 1:
        rand_ind = random.randint(0,2)
        vac_g[i][rand_ind] = 1

vac_g = pd.DataFrame(vac_g,columns = ['Vacc_1', 'Vacc_2', 'Vacc_3'])
sym_g = pd.DataFrame(sym_g,columns = ['Covid-Recovered', 'Covid-Positive', 'No-Taste/Smell', 'Fever', 'Headache', 'Pneumonia', 'Stomach', 'Myocarditis', 'Blood-Clots', 'Death'])

vacced = sym_g[np.sum(vac_g.iloc[:,-3:], axis=1) == 1]
un_vacced = sym_g[np.sum(vac_g.iloc[:,-3:], axis=1) == 0]
prior_probs_generated = [np.sum(sym_g.iloc[:,i]) / len(sym_g) for i, key in enumerate(sym_g.columns)]

pipe = Pipeline_observational(None,None,None,None)
pipe.run_efficacy(vacced, un_vacced,prior_probs_generated)




"""
vac_g = pd.DataFrame(vac_g,columns = ['Vacc_1', 'Vacc_2', 'Vacc_3'])
sym_g = pd.DataFrame(sym_g,columns = ['Covid-Recovered', 'Covid-Positive', 'No-Taste/Smell', 'Fever', 'Headache', 'Pneumonia', 'Stomach', 'Myocarditis', 'Blood-Clots', 'Death'])

vacced = sym_g[np.sum(vac_g.iloc[:,-3:], axis=1) == 1]
un_vacced = sym_g[np.sum(vac_g.iloc[:,-3:], axis=1) == 0]
prior_probs_generated = [np.sum(sym_g.iloc[:,i]) / len(sym_g) for i, key in enumerate(sym_g.columns)]

pipe = Pipeline_observational(None,None,None,None)
pipe.run_efficacy(vacced, un_vacced,prior_probs_generated)"""
"""
for i in range(len(vac_g)):
    prob = random.uniform(0,1)

    if prob > 0.6 and prob <= 0.61: #vaksinert og har covid og symptom
        rand_ind = random.randint(0,2)
        vac_g[i][rand_ind] = 1
        sym_g[i][1] = 1
        rand_sym = random.randint(0,9)
        sym_g[i][rand_sym] = 1

    if prob > 0.61 and prob <= 0.62: #vaksinert og har covid og ingen symptom
        rand_ind = random.randint(0,2)
        vac_g[i][rand_ind] = 1
        sym_g[i][1] = 1

    if prob > 0.62 and prob <= 0.75:#vaksinert og ingen symptomer
        rand_ind = random.randint(0,2)
        vac_g[i][rand_ind] = 1

    if prob > 0.75 and prob <= 0.95: #ikke vaksinert og har covid og andre symptom
        rand_sym = random.randint(0,9)
        sym_g[i][rand_sym] = 1
        sym_g[i][1] = 1

    if prob > 0.95: #ikke vaksinert og har covid og ingen andre symptom
        sym_g[i][1] = 1



vac_g = pd.DataFrame(vac_g,columns = ['Vacc_1', 'Vacc_2', 'Vacc_3'])
sym_g = pd.DataFrame(sym_g,columns = ['Covid-Recovered', 'Covid-Positive', 'No-Taste/Smell', 'Fever', 'Headache', 'Pneumonia', 'Stomach', 'Myocarditis', 'Blood-Clots', 'Death'])

vacced = sym_g[np.sum(vac_g.iloc[:,-3:], axis=1) == 1]
un_vacced = sym_g[np.sum(vac_g.iloc[:,-3:], axis=1) == 0]
prior_probs_generated = [np.sum(sym_g.iloc[:,i]) / len(sym_g) for i, key in enumerate(sym_g.columns)]

pipe = Pipeline_observational(None,None,None,None)
pipe.run_efficacy(vacced, un_vacced,prior_probs_generated)

vacced = sym_g[np.sum(vac_g.iloc[:,-3:], axis=1) == 1]
un_vacced = sym_g[np.sum(vac_g.iloc[:,-3:], axis=1) == 0]
prior_probs_generated = [np.sum(sym_g.iloc[:,i]) / len(sym_g) for i, key in enumerate(sym_g.columns)]

print(prior_probs_generated)

pipe = Pipeline_observational(None,None,None,None)
pipe.run_efficacy(vacced, un_vacced,prior_probs_generated)
"""
