# You need to tune 'threshold' so that the method is reliable.
# For example, you should define a probability with which you would like to
# include the correct feature
# and a probability with which you should not include useless features.
# It should return a list of feature indices
# This is an example select_features implementation, but you should build your
# own, preferrably through an existing feature selection method
# This implementation performs, separately for every feature, model selection
# versus the null hypothesis that Y is independent of X_i. That is, we have
# $\mu_0: P(Y|X_i) = P(Y)$
# $\mu_i: P(Y|X_i = 0) \neq P(Y)$.
# This can be implemented through n Bayesian tests.
def select_features(X, Y, threshold):
  n_features = X.shape[1]
  n_data =  X.shape[0]
  alpha_b = np.ones([n_features, 2])
  beta_b = np.ones([n_features, 2])
  log_p = np.zeros(n_features)

  log_null = 0
  alpha = 1
  beta = 1
  posterior = np.zeros(n_features)
  for t in range(n_data):
    # This is the marginal probability of Y according to model $\mu_0$
    p_null = alpha / (alpha + beta)
    # Taking the log for computational reasons
    log_null += np.log(p_null)*Y[t] + np.log(1-p_null)*(1 - Y[t])
    # Update the parameters for the null model $\mu_0$
    alpha += Y[t]
    beta += (1 - Y[t])
    # I keep a separate model \mu_i for each feature i
    for i in range(n_features):
      x_ti = X[t,i]
      # this is \mu_i(y_t = 1 | x_ti)
      p = alpha_b[i, x_ti] / (alpha_b[i, x_ti] + beta_b[i, x_ti])
      # This is the marginal probability \mu_i(y_t | x_ti)
      log_p[i] += np.log(p)*Y[t] + np.log(1-p)*(1 - Y[t])
      # There is a separate Beta model for each value of x_i
      alpha_b[i, x_ti] += Y[t]
      beta_b[i, x_ti] += (1 - Y[t])

  posterior = np.exp(log_p) / (np.exp(log_p) + np.exp(log_null))
  print(posterior)
  features = []
  for i in range(n_features):
    if posterior[i] > threshold:
      features.append(i)
  return features
