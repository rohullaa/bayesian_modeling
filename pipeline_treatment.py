import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import make_pipeline
import statsmodels.api as sm


##Loading the data:
treat_data = pd.read_csv("treatment_features.csv")
action_data = pd.read_csv("treatment_actions.csv")
outcome_data = pd.read_csv("treatment_outcomes.csv")

cols = (['Covid-Recovered', 'Covid-Positive', 'No-Taste/Smell', 'Fever', 'Headache', 'Pneumonia', 'Stomach', 'Myocarditis', 'Blood-Clots', 'Death',
        'Age', 'Gender', 'Income'] +
         [f'Gene_{i+1:03}' for i in range(128)] +
         ['Asthma', 'Obesity', 'Smoking', 'Diabetes', 'Heart disease', 'Hypertension',
         'Vacc_1', 'Vacc_2', 'Vacc_3'])

symptom_names = ['Covid-Recovered', 'Covid-Positive', 'No-Taste/Smell', 'Fever', 'Headache', 'Pneumonia', 'Stomach', 'Myocarditis', 'Blood-Clots', 'Death']
treat_data.columns = cols
outcome_data.columns = cols[:10]
action_data.columns = ['Treatment_1', 'Treatment_2']

class Pipeline_treatment():
    def __init__(self,treat_data,action_data,outcome_data):
        import warnings
        warnings.filterwarnings('ignore')

        new_treat_data = treat_data[((np.sum(treat_data.iloc[:,2:10],axis=1) > 0.0) | np.sum(outcome_data.iloc[:,2:10],axis=1) > 0.0)]
        group_first = new_treat_data[((action_data.iloc[:,0] == 1) & (action_data.iloc[:,1] == 0))]
        group_second = new_treat_data[((action_data.iloc[:,0] == 0) & (action_data.iloc[:,1] == 1))]
        group_both = new_treat_data[((action_data.iloc[:,0] == 1) & (action_data.iloc[:,1] == 1))]
        group_none = new_treat_data[((action_data.iloc[:,0] == 0) & (action_data.iloc[:,1] == 0))]

        new_outcome_data = outcome_data[((np.sum(treat_data.iloc[:,2:10],axis=1) > 0.0) | np.sum(outcome_data.iloc[:,2:10],axis=1) > 0.0)]
        outcome_first = new_outcome_data[((action_data.iloc[:,0] == 1) & (action_data.iloc[:,1] == 0))]
        outcome_second = new_outcome_data[((action_data.iloc[:,0] == 0) & (action_data.iloc[:,1] == 1))]
        outcome_both = new_outcome_data[((action_data.iloc[:,0] == 1) & (action_data.iloc[:,1] == 1))]
        outcome_none = new_outcome_data[((action_data.iloc[:,0] == 0) & (action_data.iloc[:,1] == 0))]
        prior_probs= [(np.sum(new_treat_data[sym]) + np.sum(new_outcome_data[sym])) / (len(new_treat_data) * 2) for sym in symptom_names][2:]

        for outcome_treated, pre_treated, treatment in zip([outcome_first, outcome_second, outcome_both],[group_first, group_second, group_both],['treatment 1', 'treatment 2', 'both treatments']):
            print(f"{treatment} efficacy:")
            for i, key in enumerate(outcome_data.keys()[2:]):
                #print(key)
                res = self.treatment_efficacy(outcome_treated, pre_treated, outcome_none, group_none, prior_probs[i], key)
            print()


    def find_alpha(self, beta,p):
        """ Given beta and a mean probability p, compute and return the alpha of a beta distribution. """
        return beta*p/(1-p)

    def treatment_efficacy(self, outcome_treated, precondition_treated, outcome_untreated, precondition_untreated, p, symptom_name, log=True):
        group_pos_count = np.sum(outcome_treated[symptom_name])
        group_neg_count = np.sum(outcome_untreated[symptom_name])

        group_pos_total = np.sum(precondition_treated[symptom_name])
        group_neg_total = np.sum(precondition_untreated[symptom_name])

        if any(v == 0 for v in (group_pos_total, group_neg_total, group_neg_count)):
            print(f'{symptom_name:15s}: Division by zero - not enough data to compute efficacy' )
            return

        v = group_pos_count / group_pos_total
        n_v = group_neg_count / group_neg_total
        IRR = v/n_v

        efficacy = 100 * (1- IRR)

        N = 100_000
        beta = 1
        alpha = self.find_alpha(beta,p)

        #symptom_name = symptom_names[symptom_index]
        samples_group_pos = stats.beta.rvs(alpha + group_pos_count, beta + len(outcome_treated) - group_pos_count, size=N)
        samples_group_neg = stats.beta.rvs(alpha + group_neg_count, beta + len(outcome_untreated) - group_neg_count, size=N)

        samples_ve = 100 * (1 - samples_group_pos/samples_group_neg)
        lower = np.percentile(samples_ve, 2.5)
        upper = np.percentile(samples_ve, 97.5)
        if log is True:
            print(f'{symptom_name:15s}: {efficacy:7.3f} - 95% CI: ({lower:3.3f}, {upper:3.3f})')

        return efficacy, (lower, upper)


Pipeline_treatment(treat_data,action_data,outcome_data)
