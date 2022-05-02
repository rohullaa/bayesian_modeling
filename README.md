# Bayesian Data Analysis

## Data
The first data set in the project is an observational study of a random sample.
It is contained in the file: 'observation_features.csv'.

* Symptoms (10 bits): Covid-Recovered, Covid-Positive, No-Taste/Smell, Fever, Headache, Pneumonia, Stomach, Myocarditis, Blood-Clots, Death
* Age (integer)
* Gender (binary)
* Income (floating)
* Genome (128 bits)
* Comorbidities (6 bits): Asthma, Obesity, Smoking, Diabetes, Heart disease, Hypertension
* Vaccination status (3 bits): 0 for unvaccinated, 1 for receiving a specific vaccine for each bit


The second is the result of interventions after the patients were found to be covid-positive. The patients may not necessarily have developed symptoms before the treatment was administered. The data is in three files, with each line corresponding to one patient:

* treatment_features.csv 
* treatment_action.csv
* treatment_outcome.csv

The first file is just like the observation_features file in the first data set. The other two files describe the treatments and outcomes respectively.

* Treatment (k bits): Multiple simultaneous treatments are possible 
* Post-Treatment Symptoms (10 bits): Past-Covid (Ignore), Covid+ (Ignore), No-Taste/Smell, Fever, Headache, Pneumonia, Stomach, Myocarditis, Blood-Clots, Death

## Tasks:

1. Perform the following modelling tasks for the observational variables:
   * Predicting the effect of genes and/or age/comorbidities on symptoms.
   * Estimating the efficacy of vaccines.
   * Estimating the probability of vaccination side-effects.

2. Model the effect of treatments on alleviating symptoms (e.g. preventing death). By this I mean, (a) formalising, (b) implementing and (c) verifying some type of model that can predict the effects of different treatments on symptoms. (As in case 1, you must start by specifying what relationships you wish to model, then defining the precise model you wish to use, selecting or creating an implementation, using it on the data, and verifying it works appropriately through simulation and bootstrapping/cross-validation).
