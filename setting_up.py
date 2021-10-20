import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
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

class Pipeline_A():
    def __init__(self,X,y,clf,random_state=None):

        self.X = X
        self.Y = y
        self.clf = clf
        self.threshold = threshold = 0.8
        self.random_state = random_state
        self.parameter_grid = parameter_grid = [{'kernel': ['poly', 'rbf'],
                                                'C': [0.01, 0.1,1, 10, 100,],
                                                'gamma': [.1, .01, 1e-3]}, ]

        #finding the best features:
        self.best_features = best_features = self.select_features(X,y,threshold)

        #tuning the parameters for the given clf, and evaluation the models
        print("Classification using best features")
        self.tune_parameters(X.iloc[:,best_features],y,clf,parameter_grid)

        print("Classification using all features")
        self.tune_parameters(X,y,clf,parameter_grid)

    def select_features(self, X, Y, threshold):
        """ Select the most important features of a data set, where X (2D)
        contains the feature data, and Y (1D) contains the target
        """
        X, Y = np.array(X), np.array(Y)

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
        #p = np.exp(log_p) / (np.exp(log_p) + np.exp(log_null))
        p = 1 / (np.exp(log_null - log_p) + 1)
        #print(f"{(log_p)=}\n{(log_null)=}\n{(log_p) + (log_null)=}\n {p=}")
        #print(f"{np.exp(log_p)=}\n{np.exp(log_null)=}\n{np.exp(log_p) + np.exp(log_null)=}")

        features = [i for i in range(n_features) if p[i] > threshold]

        return features

    def tune_parameters(self, X, y, clf, parameter_grid, scoring=None, cv=None):
        """ Given X, y, a classifier and a parameter grid,
        find the best parameters for the classifier and data using GridSearch
        with cross validation.
        """
        # The code below is from
        # https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=self.random_state)

        print(f"# Tuning hyper-parameters for {scoring=}")
        print()

        clf = GridSearchCV(    clf,
                                parameter_grid,
                                scoring=scoring,
                                n_jobs=-1,
                                cv=cv
                            ).fit(X_train, y_train)

        #piped_clf
        print("Best parameters set found on development set:")
        print()
        print(f"{clf.best_params_}, score: {clf.best_score_:.4f}")
        print()
        """print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()"""

        print("Classification report:")
        print()

        print(classification_report(y_test, clf.predict(X_test)))
        print()

    def train_test():
        pass
    def model_evaluation_select_features():
        pass

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
        print(f'{symptom_name:15s}: {efficacy:3.3f} - ({lower:3.3f}, {upper:3.3f})')
    def side_effects(self, vacced_neg, un_vacced_neg, start, end):
        df = pd.DataFrame(index=vacced_neg.keys()[start:end],
                          columns = ("p1 (%)", "p2 (%)", "Diff (%)", "Credible Interval (%)", "Null Hypothesis", ),
                         )

        for i in range(start, end):
            symptom = vacced_neg.keys()[i]
            p1 = vacced_neg.sum()[i] / len(self.y) / (len(vacced_neg) / len(self.y))
            p2 = un_vacced_neg.sum()[i] / len(self.y) / (len(un_vacced_neg) / len(self.y))



            lower = (p1-p2 - 1.64 * np.sqrt((p1*(1-p1) / len(vacced_neg)) + (p2 * (1-p2) / len(un_vacced_neg))))
            higher = (p1-p2 + 1.64 * np.sqrt((p1*(1-p1) / len(vacced_neg)) + (p2 * (1-p2) / len(un_vacced_neg))))

            p1, p2, lower, higher = p1 * 100, p2 * 100, lower * 100, higher * 100

            df.loc[symptom] = [round(p1, 4), round(p2, 4), round(p1 - p2, 4), (round(lower, 4), round(higher, 4)),
                               "rejected" if lower>0 else "not rejected", ]


        return df



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

symptoms = obs_data.iloc[:,0:10]
age = obs_data.iloc[:,10]
gender = obs_data.iloc[:,11]
income = obs_data.iloc[:,12]
genome = obs_data.iloc[:,13:141]
comorbidities = obs_data.iloc[:,141:147]
vaccination_status = np.array(obs_data.iloc[:,147:])


##testing the class for 1a)
pipe = Pipeline_A(genome,symptoms.iloc[:,1],SVC())


"""
#samples with vaccinations or not:
vacced = obs_data[np.sum(obs_data.iloc[:,-3:], axis=1) == 1]
vacced_neg = vacced[vacced.iloc[:,1]==0]
vacced_pos = vacced[vacced.iloc[:,1]==1]

un_vacced = obs_data[np.sum(obs_data.iloc[:,-3:], axis=1) == 0]
un_vacced_neg = un_vacced[un_vacced.iloc[:,1]==0]
un_vacced_pos = un_vacced[un_vacced.iloc[:,1]==1]

symptom_names = ['Covid-Recovered', 'Covid-Positive', 'No-Taste/Smell', 'Fever', 'Headache', 'Pneumonia', 'Stomach', 'Myocarditis', 'Blood-Clots', 'Death']
prior_probs= [np.sum(obs_data.iloc[:,i]) / len(obs_data) for i, key in enumerate(symptom_names)]

#three different vaccination types:
vacc_type1 = obs_data[obs_data.Vacc_1 == 1]
vacc_type2 = obs_data[obs_data.Vacc_2 == 1]
vacc_type3 = obs_data[obs_data.Vacc_3 == 1]
vaccination_types = [vacc_type1,vacc_type2,vacc_type3]
vaccination_names = ['type 1', 'type 2', 'type 3']
"""
