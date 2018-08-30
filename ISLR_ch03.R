# 3. Linear Regression
## snergy effect(tv ad and radio ad) :: interaction effect

# 3.1 simple linear regresssion
# Consequently it is a simple matter to compute the probability of observing any 
# value equal to |t| or larger, asumming bata_1=0. 
# We call this probability the p_value. Roughly speaking, we interpret the p_value 
# as follows: a small p-value indicates that it is unlikely to observe such a substantial 
# association between the predictor and the response due to chance, in the absence of 
# any real association between the predictor and the response.

# 3.1.3 Assessing the accuracy of the model
# residual standard error(RSE)
# the R square statistic

# 3.2 multiple linearregression

# 3.2.2. Some Important Question
# One :: Is There a Relationship between the response and predictors?

# the alternative hypothesys :: at least one beta_j is non_zero.

# when there is no relationship between the response and predictors, one would
# expect the F-stastic to take on a value close to 1.

# The large F-statistic suggest that at least one of  the advertising media must be 
# related to sales
# when n is large, an F-stastic that is just a little larger than 1 might still 
# provide evidence against H_0
# a larger F-statistic is needed to reject H_0 if n is small

# when H_0 is true and the errors epsilon_i have a normal distribution, the F-statistic
# follows an F-distribution

# when the number of variables(p) is large, some of the approachs discussed in the
# next section, such as forward selection, can be used.

# two :: deciding on important variables

# the task of determinig which predictors are associated with the response, in order to
# fit a single model involving only those predictors, is referred to as variable selection
# **chap 6 :: variable selection
# forward selection
# backward selection
# mixed selection :: the combination of forward and backward selection, the p-value for
# variables can become larger as new predictors are added to the model. Hence, if at any
# point the p-value for one of the variables in the model rises above a certain threshold, then
# we remove that variable from the model. we continue to perform these forward andn backward 
# steps until all variables in the model have a sufficiently low p-value, and all variables
# in the model have a larget p-value if added to the model

# backward selection cannot be used if p > n, while forward selection can alwas be used.
# Forward selection is a greedy approach, and might include variables early that later 
# become rednudant. Mixed selection can remedy this.

# three :: model fit
# RSE, R^2
# one property of the fitted linear model is that it maximize this correlation among
# all possible linear models
# it turns out that R^2 will always increase when more variables are added to the mdoel,
# even if those vaiables are only weakly associated with the response.

# four :: predictions

# 3.3 other considerations in the regression model

# 3.3.1 Qualitative predictors
# predictos with only two levels
# factor level, dummy variale

# we notice that the p-value for the dummy variable is very high. This indicates that
# there is no statistical evidence of a difference in average credit card balance 
# between genders.

# the p-value associated with the coefficient estimates for the two dummy variables are
# very large, suggesting no statisircal evidence of a real difference in credit card
# balance the ethnicities

# 3.3.2 extensions of linear model

# removing the additive assumption
# the effect on sales of increasing one advertising medium is independent of the amount
# spent on the other media
# in marketing, synergy effect
# in statistic, interaction effect

# the hierarchical principle states that if we include an interaction in a model, we
# should also include the main effects, even if the p-values associated with their
# coefficients are not significant.

# the concept of interactions applies just as well to qualitative variables, or to
# a combination of quantitive and qualitative variables

# this suggests that increases in income are associated with smaller increases in credit
# card balance among students as compared to non-students

# Non-linear relationship

# quadratic shape
# it is still a linear model!

# 3.3.3 potential problems
# 1. Non-linearity of the response-predictor relationships
# 2. correlation of error terms
# 3. non-constant variance of error terms
# 4. outliers
# 5. collinearity

# 1. on-linearity of the data
# residual plot

# 2. correlation of error terms
# if the error terms are correlated, we may have an unwarranted sense of confidence
# in our model
# time-series...
# In general, the assumption of uncorrelated errors is extremely important for linear
# regression as well as for other statistical methods, and good experimental design
# is crucial in order to mitigate the risk of such correlations

# 3. Non-constant variance of error terms
# heteroscedasticity, funnel shape
# log(Y), log(sqrt(Y))

# 4. outliers
# studentized residual :: Observations whose studentized residuals are greater than 3
# in absolute value are possibel outliers

# 5. high leverage points
# leverage statistic

# 6. collinearity
# collinearity is to compute the variance inflation factor(VIF)
# as a rule of thumb, a VIF value that exceeds 5 or 10 indicate a problematic amount of
# collinearity

# 3.4 the marketing plan

# 1. Is there a relationship between advertising sales annd budget?
## - p-value corresponding to the F-statistic 

# 2. How strong is the relationship?
## - RSE estimate the standard deviation of the response from the population regression line.
## - R^2 statistic records the percentage of variability in the response that is explained
##   by the predictors.

# 3. which media contribute to sales?
## - the p-value associated with each predictor's t-statistic(chap6 for details...)

# 4. How large is the effect of each medium on sales?
## - the confidence intervlas for beta

# 5. how accurately can we predict future sales?
## - prediction interval, confidence interval

# 6. Is the relationship linear?
## - residual plot

# 7. Is there synergy among the advertising media?
## - interaction effect

# 3.5 :: Comparison of Linear Regression with KNN
##  - non-parametric methods :: KNN regression

# Comparison of Linear Regression with KNN 

## - A small value for K provides the most flexible fit, which will have low bias but
## high variance. This variane is due to the fact that the prediction in a given region
## is entirely dependent on just one observation

## the parametric approach will ouitperform the non-parametric approach if the parametric
## form that has been selected is close to the true form of f.

## As a general rule, parametric method will tend to outperform non-parametric approaches
## when there is a small number of observation per predictor

# 3.6 Lab :: Linear Regression
library(MASS)
library(ISLR)

fix(Boston)
names(Boston)

lm.fit = lm(medv ~ lstat, data=Boston)
lm.fit
summary(lm.fit)

names(lm.fit)
coef(lm.fit)
# in order to obtain a confidence interval for the coefficient estimates :: confint()
confint(lm.fit)
predict(lm.fit, data.frame(lstat=(c(5,10,15))), 
        interval='confidence')
# fit      lwr      upr
# 1 29.80359 29.00741 30.59978
# 2 25.05335 24.47413 25.63256
# 3 20.30310 19.73159 20.87461 
attach(Boston)
plot(lstat, medv)
abline(lm.fit)

windows()
par(mfrow=c(2,2))
plot(lm.fit)

par(mfrow=c(1,1))
plot(predict(lm.fit), residuals(lm.fit))
plot(predict(lm.fit), rstudent(lm.fit))

plot(hatvalues(lm.fit))
which.max(hatvalues(lm.fit))

# 3.6.3 multiple linear regression
lm.fit = lm(medv ~ lstat + age, data=Boston)
summary(lm.fit)

lm.fit = lm(medv ~. , data=Boston)
summary(lm.fit)

library(car)
vif(lm.fit)

lm.fit1 = lm(medv ~.-age, data=Boston)
summary(lm.fit1)

# 3.6.4 interaction terms

summary(lm(medv ~ lstat*age, data=Boston))

# 3.6.5 Non-linear Transformation of the predictors
lm.fit2= lm(medv ~ lstat + I(lstat^2))
summary(lm.fit2)

# anova() :: this function performs a hypothesis test comparing the two models.
# The null hypothesis is that the two models fit the data equally well, and
# and the alternative hypothesis is that the fulll model is superior.

par(mfrow=c(2, 2))
plot(lm.fit2)

# poly()
lm.fit5 = lm(medv ~ poly(lstat, 5))
summary(lm.fit5)
summary(lm(medv~log(rm), data=Boston))

# 3.6.6 Qualitative Predictors
fix(Carseats)
names(Carseats)
str(Carseats)
lm.fit = lm(Sales ~.+Income:Advertising + Price:Age, data=Carseats)
summary(lm.fit)
attach(Carseats)
contrasts(ShelveLoc) # return the coding that R uses for the dummy variables

# ShelveLocGood in the regression output is positive indicates that a good shelving 
# location is associated with high sales

LoadLibraries=function() {
  library(ISLR)
  library(MASS)
  print('The libraries have been loaded')
}

LoadLibraries
LoadLibraries()

# 3.7 Exercise

# 1.
# TV :: p_value = .0001
# radio : p_value = .0001
# newspaper : P_value = .8599

# TV and radio are related to sales but no evidence that newspaper is associated with sales
# in the presence of other predictors

# 2.
# KNN regression averages the closest observations to estimate prediction, KNN classifier
# assigns classification group based on majority of closest observations

# 3.
# Y = 50 + 20*GPA + .07*IQ + 35*Gender + .01*GPA:IQ - 10*GPA:Gender
# For GPA above 35/10=3.5, males will earn more.
# (Gender :: 1 for woman, 0 for man)

# 50 + 20*4 + .07*110 + 35*1 + .01*4*110 -10*4*1
# $ 137100

# False
# IQ scale is larger than other predictors (~100 versus 1-4 for GPA and 0-1 for gender) so 
# even if all predictors have the smae impact on salary, coefficients will be smaller for
# IQ predictors

# 4.
# (a) Having more predictors generally means better(lower) RSS on training data
# (b)If the additional predictors lead to overfitting, the testing RSS could be worse(higher)
# for the cubic regerssion fit
# (c) The cubic regression fit should produce a better RSS on the training set because
# it can adjust for the non-linearity
# (d) Similer to trainig RSS, the cubic regression fit shoud produce a better RSS on the
# testing set because it can adjust for the non;linearity

# 5.

# 6. 

# 7. 

# 8. 
data(Auto)
fit.lm <- lm(mpg ~ horsepower, data=Auto)
summary(fit.lm)

# 1) yes, there is a relationship between predictor and response
# 2) p-value is close to 0, relationship is strong
# 3) Coefficient is negative :  relationship is negative
# 4) 

new <- data.frame(horsepower=98)
predict(fit.lm, new)
predict(fit.lm, new, interval = 'confidence')
predict(fit.lm, new, interval = 'prediction')

windows()
plot(Auto$horsepower, Auto$mpg)
abline(fit.lm, col='red')

par(mfrow=c(2,2))
plot(fit.lm)
par(mfrow=c(1,1))

# residuals vs fitted plot shows that the relationship is non-linear

# 9.
# part a)
data(Auto)
pairs(Auto)

# part b)
str(Auto)
subset(Auto, select=-name)
head(subset(Auto, select=-name))
cor(subset(Auto, select=-name))

# part c)
fit.lm <- lm(mpg ~.-name, data=Auto)
summary(fit.lm)
# - There is a relationship between predictors andn response
# - weight, year, origin and displacement have statistically significant relationships
# - .75 coefficient for year suggests that later model year cars have better(higher) mpg


















































