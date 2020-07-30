library(tidyverse)
library(assertr)
library(readxl)

d<-read_excel('data/Question 1 - AB Testing Data.xlsx')

#Check that each member_id has only one variant label

d %>% 
  group_by(member_id) %>% 
  summarise(num_labs = n_distinct(variant)) %>% 
  verify(num_labs==1) #One variant per member verification

# Check to see if each member posts one job
# False.  Some members post multiple jobs
# Violation of the iid principle
d %>% 
  group_by(member_id) %>% 
  summarise(n_job = n_distinct(job_id)) %>% 
  arrange(desc(n_job)) %>% 
  verify(max(n_job)>1)
  

# Check to see if there is a unique Hired label
# for each job id
# True
d %>% 
  group_by(member_id, job_id) %>% 
  summarise(nHired = n_distinct(Hired)) %>% 
  verify(nHired==1)


# Ok, to examine the effects of the ab-test, we can filter to
# member_id, job_id, variant, Hired

ab_data <- select(d, member_id, job_id, variant, Hired) %>% 
          distinct(member_id, job_id, variant, Hired) %>% 
          group_by(member_id, variant) %>% 
          summarise(y=sum(Hired), n = n())

write_csv(ab_data, 'data/ab_data.csv')


d %>% 
  select(member_id, job_id, variant, Hired) %>% 
  arrange(member_id, job_id) %>% 
  distinct(member_id, job_id, variant)
