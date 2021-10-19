import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler


cols = (['Covid-Recovered', 'Covid-Positive', 'No-Taste/Smell', 'Fever', 'Headache', 'Pneumonia', 'Stomach', 'Myocarditis', 'Blood-Clots', 'Death',
        'Age', 'Gender', 'Income'] +
         [f'Gene_{i+1:03}' for i in range(128)] +
         ['Asthma', 'Obesity', 'Smoking', 'Diabetes', 'Heart disease', 'Hypertension',
         'Vacc_1', 'Vacc_2', 'Vacc_3'])
obs_data = pd.read_csv("observation_features.csv")
treat_data = pd.read_csv("treatment_features.csv")
action_data = pd.read_csv("treatment_actions.csv")
outcome_data = pd.read_csv("treatment_outcomes.csv")

obs_data.columns = cols
treat_data.columns = cols
outcome_data.columns = cols[:10]
action_data.columns = ['Treatment_1', 'Treatment_2']


symptoms = np.array(obs_data.iloc[:,0:10])
age = obs_data.iloc[:,10]
gender = obs_data.iloc[:,11]
income = obs_data.iloc[:,12]
genome = obs_data.iloc[:,13:141]
comorbidities = obs_data.iloc[:,141:147]
vaccination_status = np.array(obs_data.iloc[:,147:])

def select_features(X, Y, threshold):
    n_features = X.shape[1]
    n_data =  X.shape[0]
    alpha_b = np.ones([n_features, 2 ])
    beta_b = np.ones([n_features, 2])
    log_p = np.zeros(n_features)

    log_null = 0
    alpha = 1
    beta = 1
    for t in range(n_data):
        p_null = alpha / (alpha + beta)
        log_null += np.log(p_null)*Y[t] + np.log(1-p_null)*(1 - Y[t])
        alpha += Y[t]
        beta += (1 - Y[t])
        for i in range(n_features):

                x_ti = int(X[t,i])
                p = alpha_b[i, x_ti] / (alpha_b[i, x_ti] + beta_b[i, x_ti])
                log_p[i] += np.log(p)*Y[t] + np.log(1-p)*(1 - Y[t])
                alpha_b[i, x_ti] += Y[t]
                beta_b[i, x_ti] += (1 - Y[t])
    log_max=np.mean(log_p)
    log_max2=np.mean(log_null)
    log_p=log_p-log_max
    log_null=log_null-log_max2
    p = np.exp(log_p) / (np.exp(log_p) + np.exp(log_null))
    #print((np.exp(log_p) + np.exp(log_null)))

    features = []
    for i in range(n_features):
        if p[i] > threshold:
            features.append(i)
    return features

def count_status(vac_list, symptom_name):
    index = symptom_names.index(symptom_name)
    vacc_pos = 0
    for i in vac_list:
        if symptoms[i][1] == 1 and symptoms[i][index] == 1:
            vacc_pos += 1
    return vacc_pos

def find_alpha(beta,p):
    return beta*p/(1-p)

def find_efficacy(vaccinated,not_vaccinated,symptom_name,count_status):
    vacc_pos = count_status(vaccinated,symptom_name)
    not_pos = count_status(not_vaccinated,symptom_name)

    v = vacc_pos/len(vaccinated)
    n_v = not_pos/len(not_vaccinated)
    IRR = v/n_v

    efficacy = 100*(1- IRR)

    N = len(obs_data)
    beta = 1
    p = prior_probs[symptom_name]
    alpha = find_alpha(beta,p)

    samples_vaccine = stats.beta.rvs(alpha + vacc_pos, beta + len(vaccinated) - vacc_pos, size=N)
    samples_no_vaccine = stats.beta.rvs(alpha + not_pos, beta + len(not_vaccinated) - not_pos, size=N)

    samples_ve = 100 * (1 - samples_vaccine/samples_no_vaccine)
    lower = np.percentile(samples_ve, 2.5)
    upper = np.percentile(samples_ve, 97.5)
    print(f'{symptom_name:15s}: {efficacy:3.3f} - ({lower:3.3f}, {upper:3.3f})')

not_vaccinated = [i for i in range(len(vaccination_status)) if all(v==0 for v in vaccination_status[i])]
vaccinated = [i for i in range(len(vaccination_status)) if any(v !=0 for v in vaccination_status[i])]
symptom_names = ['Covid-Recovered', 'Covid-Positive', 'No-Taste/Smell', 'Fever', 'Headache', 'Pneumonia', 'Stomach', 'Myocarditis', 'Blood-Clots', 'Death']
prior_probs= {key:np.sum(obs_data.iloc[:,i]) / len(obs_data) for i, key in enumerate(symptom_names)}

def run_efficacy():
    print("Efficacy of the vaccinated:")
    for s in symptom_names:
        find_efficacy(vaccinated,not_vaccinated,s,count_status)

def run_select_features():
    s_0 = select_features(np.array(genome),6,0.8)
    print(s_0)


pipeline = Pipeline(steps= [('select_features',run_select_features()), ('run_efficacy',run_efficacy())])
print(pipeline)
