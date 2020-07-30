library(rstan)
library(tidyverse)
library(brms)
ab_data = read_csv('data/ab_data.csv')

stan_code="
data{
  int N;
  int trials[N];
  int Y[N];
  matrix[N,2] X;
}
parameters{
  real Intercept;
  real variantB;
  vector[N] z;
  real<lower=0> sigma;
}
transformed parameters{
 
 vector[2] beta=to_vector({Intercept, variantB});
 vector[N] eta = X*beta + z*sigma;
}
model{
  Intercept~normal(0.61, 0.12);
  variantB~normal(0,1);
  z~normal(0,1);
  sigma~student_t(3,0,2.5);
  Y~binomial_logit(trials, eta);
}
"

data = make_standata(y|trials(n) ~ variant + (1|member_id), 
                     data = ab_data, 
                     family  = binomial())


stan_model = stan_model(model_code = stan_code)

fit=sampling(stan_model, data = data[c('X','trials','Y','N')], pars = c('Intercept','variantB'))

p = rstan::extract(fit)

a = plogis(p$Intercept)
b = plogis(p$Intercept + p$variantB)

mean(a)
mean(b)
