import pandas as pd
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

data = pd.read_csv("https://raw.githubusercontent.com/dustywhite7/Econ8310/master/AssignmentData/cookie_cats.csv")
display(data)

# Converting retention columns to integers
data['retention_1'] = data['retention_1'].astype(int)
data['retention_7'] = data['retention_7'].astype(int)

# Separation data for control and treatment groups
control_1day = data[data['version'] == 'gate_30']['retention_1']
treatment_1day = data[data['version'] == 'gate_40']['retention_1']

# Bayesian model for 1-day retention
with pm.Model() as model_1day:
    # Priors for probabilities
    p_control = pm.Beta('p_control', alpha=1, beta=1)
    p_treatment = pm.Beta('p_treatment', alpha=1, beta=1)

    # Likelihood
    control_obs = pm.Binomial('control_obs', n=len(control_1day), p=p_control, observed=control_1day.sum())
    treatment_obs = pm.Binomial('treatment_obs', n=len(treatment_1day), p=p_treatment, observed=treatment_1day.sum())

    # Difference in probabilities
    difference = pm.Deterministic('difference', p_treatment - p_control)

    # Sample from posterior
    trace_1day = pm.sample(2000, return_inferencedata=True)

# Summary of results for 1-day retention
summary_1day = az.summary(trace_1day, hdi_prob=0.95)
print(summary_1day)

# Control and treatment definition
control_7day = data[data['version'] == 'gate_30']['retention_7']
treatment_7day = data[data['version'] == 'gate_40']['retention_7']

# Bayesian model for 7-day retention
with pm.Model() as model_7day:
    # Priors for probabilities
    p_control7 = pm.Beta('p_control7', alpha=1, beta=1)
    p_treatment7 = pm.Beta('p_treatment7', alpha=1, beta=1)

    # Likelihood
    control7_obs = pm.Binomial('control7_obs', n=len(control_7day), p=p_control7, observed=control_7day.sum())
    treatment7_obs = pm.Binomial('treatment7_obs', n=len(treatment_7day), p=p_treatment7, observed=treatment_7day.sum())

    # Difference in probabilities
    difference7 = pm.Deterministic('difference7', p_treatment7 - p_control7)

    # Sample from posterior
    trace_7day = pm.sample(2000, return_inferencedata=True)

# Summary of results for 7-day retention
summary_7day = az.summary(trace_7day, hdi_prob=0.95)
print(summary_7day)

# Visualize Results
# Posterior plot for 1-day retention difference
az.plot_posterior(trace_1day, var_names=['difference'], hdi_prob=0.95)
plt.title("Posterior Distribution of 1-Day Retention Difference")
plt.show()
# Posterior plot for 7-day retention difference
az.plot_posterior(trace_7day, var_names=['difference7'], hdi_prob=0.95)
plt.title("Posterior Distribution of 7-Day Retention Difference")
plt.show()
