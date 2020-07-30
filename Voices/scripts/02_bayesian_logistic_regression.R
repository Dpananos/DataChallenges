library(brms)
library(tidyverse)
library(here)
library(tidybayes)
ab_data = read_csv('data/ab_data.csv')

#Set prior.  Prior parameters were determined by
#Root finding procedures to ensure the 2.5% and 97.5% quantiles were 0.6 and 0.6
baseline_prior<-set_prior('normal(0.61, 0.12)', class = 'Intercept')
variant_prior<-set_prior('normal(0,1)', class = 'b', coef = 'variantB')

#Model the data, specifying a random effect for the members
#Model diagnostics do not indicate patholigical behaviour, 
#Good for inference!
model = brm(y|trials(n) ~ variant + (1|member_id), 
            data = ab_data, 
            prior = baseline_prior + variant_prior,
            family  = binomial())



a = posterior_linpred(model, 
                      re_formula =NA, 
                      newdata = tibble(n=1, variant = 'A'))

b = posterior_linpred(model, 
                      re_formula =NA, 
                      newdata = tibble(n=1, variant = 'B'))

mean(plogis(a))
mean(plogis(b))
