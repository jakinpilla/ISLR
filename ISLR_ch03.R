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
new
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
# subset(Auto, select=-name)
head(subset(Auto, select=-name))
cor(subset(Auto, select=-name))

# part c)
fit.lm <- lm(mpg ~.-name, data=Auto)
summary(fit.lm)
# - There is a relationship between predictors and response
# - weight, year, origin and displacement have statistically significant relationships
# - .75 coefficient for year suggests that later model year cars have better(higher) mpg

# part d)
par(mfrow=c(2,2))
plot(fit.lm)
# - evidence of non-linearity
# - observarion 14th has high leverage

# part e)
fit.lm_0 <- lm(mpg ~ displacement + weight + year + origin, data=Auto)
fit.lm_1 <- lm(mpg ~ displacement*weight+ year + origin, data=Auto)
fit.lm_2 <- lm(mpg ~ displacement*year + weight + origin, data=Auto)
fit.lm_3 <- lm(mpg ~ displacement*origin + weight + origin, data=Auto)
fit.lm_4 <- lm(mpg ~ displacement + weight*year + origin, data=Auto)
fit.lm_5 <- lm(mpg ~ displacement + weight*origin + year, data=Auto)
fit.lm_6 <- lm(mpg ~ displacement + weight + year*origin, data=Auto)

summary(fit.lm_1) ## displacement:weight / there is interaction effect 
summary(fit.lm_2) ## displacement:year / there is interaction effect
summary(fit.lm_3) ## displacement:origin / there is no interaction effect
summary(fit.lm_4) ## weight:year / there is interaction effect
summary(fit.lm_5) ## weight:orign / there is interaction effect
summary(fit.lm_6) ## year:origin / there is interaction effect

fit.lm <- lm(mpg ~ displacement + weight + year + origin + 
     displacement:weight + displacement:year + weight:year + weight:origin +
     year:origin, data=Auto)

summary(fit.lm)

fit.lm <- lm(mpg ~ year + displacement:weight, data=Auto)
summary(fit.lm)


fit.lm <- lm(mpg ~ displacement + weight + year + displacement:weight, data=Auto)
summary(fit.lm)

par(mfrow=c(1,1))
plot(Auto$displacement, Auto$mpg)

# part f)
# try 3 predictor transformations

fit.lm4 <- lm(mpg ~  poly(displacement, 3) + weight + year + origin, data=Auto)
summary(fit.lm4)
# displacement^2 has a larger effect than other displacement polynomials

fit.lm5 <- lm(mpg ~ displacement + I(log(weight)) + year + origin, data=Auto)
summary(fit.lm5)

fit.lm6 <- lm(mpg ~ displacement + I(weight^2) + year + origin, data=Auto)
summary(fit.lm6)

# 10.

# part a)
library(ISLR)
data("Carseats")
fit.lm <- lm(Sales ~ Price + Urban + US, data=Carseats)
summary(fit.lm)

# part b)
# Price(-0.054459) :: Sales drop by 54 for each dollar increase in Price - statistically significant
# UrbanYes(-0.021916) :: Sales are 22 lower for Urban locations - not statistically significant
# USYes(1.200573) :: Sales are 1,201 higher in the US locations - statistically significant

# part c)
# Sales = 13.043 - 0.054*Price - 0.022*UrbanYes + 1.201*USYes

# part d)
# we can reject null hypothesis for Price and USYes(coefficients have low p-value)

# part e)
fit.lm1 <- lm(Sales ~ Price + US, data=Carseats)
summary(fit.lm1)

# part f)
# fit.lm(Price, Urban, US)
# RSE = 2.472
# R^2 = 0.2393

# fit.lm1(Price, US)
# RSE = 2.469
# R^2 = .2393

# fit.lm1 has a slightly better :: lower RSE value and less predictor variable

# part g)
confint(fit.lm1)

# part h)
par(mfrow=c(2,2))
# residuals vs fitted plot doesn't show strong outliers
plot(fit.lm1)

par(mfrow=c(1,1))
plot(predict(fit.lm1), rstudent(fit.lm1))
# studentized residuals within -3 to 3 range

library(car)
qqPlot(fit.lm1, main='QQ Plot')
leveragePlots(fit.lm1) # leverage plots
plot(hatvalues(fit.lm1))

# average obs leverage (p+1)/n = (2+1)/400 = 0.0075
# data may have some leverage issues

# 11.
set.seed(1)
x=rnorm(100)
y=2*x + rnorm(100)

# part a)
fit.lm_y <- lm(y~x+0)
summary(fit.lm_y)
# small std. error for coefficient relative to coefficient estimate.
# p-value is close to zero so statistically significant

# part b)
fit.lm_x <- lm(x~y+0)
summary(fit.lm_x)
# small std. error for coefficient relative to coefficient estimate.
# p-value is close to zero so statistically significant

# part c)

# part d) 
# t-statistic
sqrt(length(x)-1)*sum(x*y) / sqrt(sum(x^2)*sum(y^2) - sum(x*y)^2)
# 18.72593

# part e) 
# Given the result of (d), notwithstanding we change x with y, t-statistic remains
# same for the regression of x onto y, comparing to the regression of y onto x.

# part f)
lm.fit_y_onto_x = lm(y~x) # with an intercept
summary(lm.fit_y_onto_x)

lm.fit_x_onto_y = lm(x~y)
summary(lm.fit_x_onto_y)

# t-statistic for above two models are same(18.56)

# 12.

# part a)
# when sum(x_i)^2 = sum(y_i)^2, the coefficient estimate for the regression of X
# onto Y is the same as the one for regression of Y on X

# part b)
set.seed(1)
x=rnorm(100)
y=2*x

sum(x)
sum(y)

lm.fit1 = lm(y ~ x + 0)
lm.fit2 = lm(x ~ y +0)

summary(lm.fit1) # 2.0
summary(lm.fit2) # .5

# the regression coefficient estimates are different for above two models

# part c)
set.seed(1)
x=rnorm(100)
y=sample(x, replace=FALSE, 100) # a random permutation
sum(x^2) == sum(y^2)

lm.fit1 = lm(y ~ x + 0)
lm.fit2 = lm(x ~ y + 0)

summary(lm.fit1)
summary(lm.fit2)

# the regression coefficient estimates are same for above two models

# 13.

# part a)
set.seed(1)
x = rnorm(100)

# part b)
eps = rnorm(100, mean=0, sd=.25)

# part c)
y = -1 + .5*x + eps
length(y)
# beta_0= -1, beta_1 = .5

# part d)
plot(x, y)
# x and y has a positive correlation

# part e)
lm.fit <- lm(y ~ x)
summary(lm.fit)
# almost same

# part f)
plot(x, y)
abline(-1, .5, lwd=2, col='blue', lty=2)
abline(lm.fit, lwd=2, col='red', lty=1)
legend('bottomright',
       c('population regression line', 'least squares line'),
       lty=c(2, 1), lwd=c(2, 2), col=c('blue', 'red'))

# part g)
lm.fit2 <- lm(y ~ x + I(x^2))
summary(lm.fit2)

anova(lm.fit, lm.fit2)

# R^2 remain the same (.7828) for the two models, therefore there is no
# evidence that the quadratic term improves the model fit

# part h) 
set.seed(1)
x = rnorm(100)
eps = rnorm(100, mean=0, sd=.1)
y = -1 + .5*x + eps
lm.fit3 <- lm(y~x)

plot(x, y)
abline(-1, .5, col='blue', lty='dashed')
abline(lm.fit3, col='red')
legend('bottomright', 
       c('population regression line', 'least squares line'),
       lty=c(2,1), lwd=c(2.5, 2.5), col=c('blue', 'red'))

summary(lm.fit3)
# R^2 has been increased from .7784 to .9565 indicating that the model fits current
# dataset better

# part 13)
set.seed(1)
x = rnorm(100)
eps = rnorm(100, mean=0, sd=.5)
y=-1 + .5*x + eps

lm.fit4 = lm(y ~ x)

windows()
plot(x, y)
abline(-1, .5, lwd=2, col='blue', lty='dashed')
abline(lm.fit4, lwd=2, col='red')
legend('bottomright', 
       c('population regression line', 'least squares line'),
       lty=c(2,1), lwd=c(2.5, 2.5), col=c('blue', 'red'))

# part j)

# original dataset
confint(lm.fit)
# less noisy dataset
confint(lm.fit3)
# noiser dataset
confint(lm.fit4)

# 14.
# part a)
set.seed(1)
x1 = runif(100)
x2 = .5*x1 + rnorm(100)/10
y = 2 + 2*x1 + .3*x2 + rnorm(100)

# part b)
plot(x1, x2)

# part c)
lm.fit = lm(y ~ x1 + x2)
summary(lm.fit)

# Due to p-value for H_0 : hat_bata_1 = 0, which is less than cutoff value .05,
# therefore we can reject the null hypothesis and favor the alternative one.

# P-value for H_0 : hat_beta_2 = 0 is 0.3754, which is much greater than cutoff
# value, therefore there is not enough to reject H_0 : hat_beta_2 = 0

# part d)

lm.fit2 = lm(y ~ x1)
summary(lm.fit2)
# p_value for hat_beta_1 = 0 is nearly 0, so we can reject the null hypothesis

# part e)
lm.fit3 = lm(y ~ x2)
summary(lm.fit3)
# p_value for hat_beta_2 = 0 is nearly 0, so we can reject the null hypithesis

# part f) 
# No, the result obtained so far do not contradict each other. Since there is
# an interaction effect between x_1 and x_2, increase any of them will increase
# the other variable and thus help fit the data. Therefore, when we do a linear 
# regression on x_1 and x_2 individually, we can't reject null hypothesis (i.e.
# we can't tell which variable indeed has no relationship with y). But y is
# regressed upon both x_1 anc x_2, in this case, we are able to reject one of them

# part g)
summary(x1)
summary(x2)
x1 = c(x1, .1)
x2 = c(x2, .8)
y = c(y, 6)

par(mfrow=c(2,2))
lm.fit4 = lm(y ~ x1 + x2)
summary(lm.fit4)

windows()
par(mfrow=c(2,2))
plot(lm.fit4)

lm.fit5 = lm(y ~ x1)
summary(lm.fit5)
plot(lm.fit5)

lm.fit6 = lm(y ~ x2)
summary(lm.fit6)
plot(lm.fit6)

# After adding new observation in x_1, the R^2 has been decreased from .2024 to .1562, 
# meaning the newly added observation is an outlier. Futhurmore, the second plots also
# show that the newly added obsevation in x_1 is an outlier

# The third plot shows that the newly added observation in x_2 is a high leverage point

# 15.

# part a)
library(ISLR)
library(MASS)
data(Boston)

attach(Boston)

names(Boston)
str(Boston)

names(Boston)[-1]
t(subset(Boston, select='zn'))
c(t(subset(Boston, select='zn')))

sim_beta_js = c()
for (name in names(Boston)[-1]) {
  predictor = c(t(subset(Boston, select = name)))
  lm.fit = lm(crim ~ predictor)
  sim_beta_js <- c(sim_beta_js, coef(lm.fit)[2])
  print(paste('Runnning simple linear regression : ', name))
  print(summary(lm.fit))
}
# All predictors except chas are statistical significant

# part b)
lm.fit = lm(crim~., data=Boston)
summary(lm.fit)

# we can reject null hypothesis for predictors including zn, dis, rad, black, medv

# part c)
# Results in (b) have much more predictors which are not statistically significant 
# comparing to the reult in (a)
sim_beta_js # :: univariate regression coefficients
coef(lm.fit)[-1] # :: multivariate regression coefficients

par(mfrow=c(1,1))
plot(sim_beta_js, coef(lm.fit)[-1])

names(Boston)
which.max(sim_beta_js)
names(Boston)[which.max(sim_beta_js) + 1] ## +1 because of 'crim' column
coef(lm.fit)[which.max(sim_beta_js) + 1]
max(sim_beta_js)

# predictor nox has univariable regression coefficient estimate of 31 and
# multiple regression coefficient estimate of -10

for (name in names(Boston)[-1]){
  predictor = c(t(subset(Boston, select=name)))
  lm.fit = lm(crim ~ predictor + I(predictor^2) + I(predictor^3)) # adding non-linearity
  print(paste('Running simple linear regression on:', name))
  print(summary(lm.fit))
}

# There are evidences of non-linear association between predictor and response 
# for indux, nox, dis, ptratio, medv.

















































































