
printf "\n#######################################################################\n"
printf "Reproducing"
printf "\n#######################################################################\n"


rm -v -- data/ab_data.csv
rm -v -- models/bayesian_ab_test.RDS


R --quiet --vanilla < scripts/01_data_cleaning.R
R --quiet --vanilla < scripts/02_bayesian_logistic_regression.R
R --quiet --vanilla < scripts/knit_report.R
rm *.tex