setwd("C:/Users/Daniel/ISLR")

# wage(salary) data :: age + year + education
# stock market data :: smartket data
# gene expression data :: NCI60 dataset, clustering, principal component

# a brief history of Statistical learning
# least squares :: linear regression
# Fisher :: linear discriminant analysis :: logistic regression
# Nelder and Wedderburn :: generalized linear models
# Breiman, Friedman...:: classification and regression tree
# Hastie, Tibshirani :: generalized additive models

# Notation and Simple Matrix Algebra
# n :: the number of distinct data points
# p :: the number of variables 
# wage data set consist of 12 varibables for 3,000 people :: n=3000, p=12
# Xij :: the value of jth variable for the ith observation
# Xi :: a vector of length p, containing the p variable measurements for ith observation
# T :: transpose

# Statiscal Learning
# Advertising data, sales, TV, radio, newspaper
# input variable :: X1=TV budget, X2=radio budget, X3=newspaper budget
# output variable :: sales
# why estimate f? :: prediction and inference
# reducible error, irreducible error
# which predictors are associated with the response?
# what is the relationship between the response ad each predictor?
# can the relationship between Y and each predictor be adequately summarized using a linear equation?
# prediction setting, inference setting, combination of the two
# the goal is to identify individuals who will respond positively to a mailing 
# based on observations of demographic variables
# what effect will changing the price of a product have on sales?

# parametric methods
## 1. we make an assumption about the functional form, or shape, of f
## 2. we need a procedure that uses the training data to fit or train the model(least squares)

# non-parametic methods

# the trade-off between prediction accuracy and model interpretability
# GAM :: generalized additive model

# supervised versus unsupervised learning

# in a market segmentation study we
# might observe multiple characteristics (variables) for potential customers,
# such as zip code, family income, and shopping habits. We might believe
# that the customers fall into different groups, such as big spenders versus
# low spenders. If the information about each customer’s spending patterns
# were available, then a supervised analysis would be possible. However, this
# information is not available—that is, we do not know whether each potential
# customer is a big spender or not. In this setting, we can try to cluster
# the customers on the basis of the variables measured, in order to identify
# distinct groups of potential customers. Identifying such groups can be of
# interest because it might be that the groups differ with respect to some
# property of interest, such as spending habits.

# regression vs classification problems
# l. Some statistical methods, such as K-nearest neighbors,
# (Chapters 2 and 4) and boosting (Chapter 8), can be used in the case of
# either quantitative or qualitative responses.

# assessing model accuracy
# there is no free lunch in statistics

# 2.2.1 measuring the quality of fit
# mse(mean squared error)
# we are interested in the accuracy of the predictions that we obtain when we
# apply our mothod to previously unseen test data
# we want to choose the mothod that gives the lowest test MSE, as opposed to the lowest traing MSE

# the bias variance trade-off
# it is possible to show that the expected test mse, for a given value X0, can always be
# decomposed into the sum of three fundamental quantities
# 1. variance of hat_f(X0)
# 2. the squared bias of hat_f(X0)
# 3. the variance of the error terms(epsilon)
# In order to mnimize the expected test error, we need to select a statistical learning
# method that simultaneously achieves low variance and low bias
# Variance refers to the amount by which ˆf would change if we estimated it using a different training data set. 
# Bias refers to the error that is introduced by approximating a real-life problem, 
# which may be extremely complicated, by a much simpler model

# 2.2.3 the classification setting
# I(Yi != ^Yi) :: if Yi=^Yi, 0 :: if Yi != ^Y, 1
# the bayes classifier
# conditional probability
# Bayes decision boundary
# KNN
# for real data, we do not know the conditional distribution of Y given X, so computing 
# the Bayes classifier is impossible.
# the Bayes classifier serves as an unattainable gold standard against which to compare other methods

rm(list=ls())
gc()
x = matrix(c(1,2,3,4), 2,2,)
x
matrix(c(1,2,3,4), 2,2, byrow=T)
sqrt(x)
x^2
x = rnorm(50) # mean=0, sd=1
mean(x)
sd(x)
y = x + rnorm(50, mean=50, sd=.1)
mean(y)
sd(y)
cor(x,y)
set.seed(1303)
rnorm(50)

x=rnorm(100)
y=rnorm(100)
plot(x, y)
pdf('Figure.pdf')
plot(x, y, col='green')
dev.off() # indicate to R that we are done creating the plot
x=seq(-pi, pi, length=50)
y=x
f = outer(x, y, function(x, y)cos(y)/(1+x^2))
contour(x, y, f)
contour(x, y, f, nlevels=45, add=T)
fa=(f-t(f))/2
contour(x,y,fa,nlevels=16)

image(x, y, fa)
persp(x, y, fa)
persp(x, y, fa, theta=30)
persp(x, y, fa, theta=30, phi=20)
persp(x, y, fa, theta=30, phi=70)

Auto = read.table('./data/Auto.data.txt', header=T)
head(Auto)
fix(Auto)

Auto = read.table("./data/Auto.data.txt", header=T, na.strings = '?')
# fix(Auto)
dim(Auto)
Auto[1:4, ]
Auto=na.omit(Auto)
dim(Auto)
names(Auto)

plot(Auto$cylinders, Auto$mpg)
attach(Auto)
plot(cylinders, mpg)
cylinders = as.factor(cylinders)
plot(cylinders, mpg) 
# if the variable plotted on the x-axis is categorical, then boxplots
# will automatically be produced by the plot()

plot(cylinders, mpg)
plot(cylinders, mpg, col='red')
plot(cylinders, mpg, col='red', varwidth=T)
plot(cylinders, mpg, col='red', varwidth=T, horizontal=T)
plot(cylinders, mpg, col='red', varwidth=T, xlab='cylinders', ylab='MPG')

hist(mpg)
hist(mpg, col=2)
hist(mpg, col=2, breaks=15)

pairs(Auto)
pairs(~ mpg + displacement + horsepower + weight + acceleration, Auto)

plot(horsepower, mpg)
identify(horsepower, mpg, name)

summary(Auto)
summary(mpg)

# 2.4 Exercise
# 1_a :: n : large, p : small :: flexible method is better
# because sample suze is large enough to fit more parameters and
# small number of predictors limits model variance

# 1_b :: n : small, p : large ::  flexible learning is worse
# because it would bt more likely to overfit

# 1_c :: the relationship between the predictors and response is highly nonlinear
# flexible is better because it is less restrictive on the shape of fit

# 1_d :: the variance of the error is high :: flexible is worse 
# because it would be more likely to overfit

# 2_a :: regression, inference, n=500, p=3(profit, number of employee, industry)
# 2_b :: classification, prediction, n= 20, p=13
# 2_c :: regression, prediction, n=52, p=3

# 3_a :: 



pdf('variance_bias_testerror.pdf')
curve(82*x, from=0, to=10, xlab='flexibility', ylab='MSE', col='white')
curve(300*cos(x/3)+500+x^3/3, add=T, col='red', lwd=2) # test error
curve(x^3/3, add=T, col='orange', lwd=2) # variance
curve(0*x + 250, add=T, col='gray', lwd=2) # irreducible error
curve(300*cos(x/3)+350, add=T, col='green', lwd=2) # bias
curve(255*cos(x/3)+450, add=T, col='blue', lwd=2) # train error
dev.off()

# 3_b
# variance will increase with higher flexibility because changing data points 
# will have more effect on the parameter estimate

# bias will decrease with higher flexibility because there are fewer assumptions
# made about the shape of the fit

# test error will have a U-shaped curve because it reflects the interaction berween 
# variance and bias

# irreducible error is the same regardless of model fit

# train error will always decrease with more model flexibility because an overfit
# model will produce lower MNE on the training data

# 4_a
# win/loss
## response :: team win or loss
## predictors :: team stength/weakness, opponent strength/weakness, player injuries

# 4_b
## salary for job posting
## response :: salary
## predictors :: position title, location, compacy/peer salaries

# 4_c
## customer segments

# 5
## Advantage of a very flexible model include better fit to data and fewer
## prior assumption. 
## Disadvantages are hard to interpret and prone to overfitting
## A more flexible approach might be preferred is the underlying data is very
## complex or if we mainly care about th result and not inference
## A less flexible model is preferred is the underlying data has a simple shape
## or if inference is important

# 6
## For parametirc methods, we make an assumption about the shape of the underlying data,
## select a model form, and fit the data to our selected form.
## The advantage here is that we can incorporate any prior/expert knowledge and 
## don't tend to have too many parameters that need to be fit.
## To the extent that our pior/expert assumption are wrong, then would a disadvantage

## None-parametric methods don't make explicit assumptions on the shape of the data. 
## This can have the advantage of not needing to make an assumption on the form of the
## function and can more accurately fit a wider range of shapes for the underlying data.
## The key disadvantage is that they need a large number of ovservations to fit accurate estimate.

# 7

x_1 = c(0, 0, 2, 0, 0, -1, 1)
x_2 = c(0, 3, 0, 1, 1, 0, 1)
x_3 = c(0, 0, 0, 3, 2, 1, 1)
y = c('red', 'red', 'red', 'green', 'green', 'red')

data <- cbind(x_1, x_2, x_3)
dist(data)

x_1 = c(0, 2, 0, 0, -1, 1)
x_2 = c(3, 0, 1, 1, 0, 1)
x_3 = c(0, 0, 3, 2, 1, 1)
df <- data.frame(x_1, x_2, x_3, y)
str(df)

# 7_b
library(class)
knn(df[, -4], c(0,0,0), df[, 4], k=1)

# 7_c
knn(df[, -4], c(0,0,0), df[, 4], k=3)

# 7_d
## best value of K shoud beb able to capture more of the non-linear decision boundary

# 8(Applied)

# install.packages('ISLR')
library(ISLR)

data(College)
str(College)

college <- read.csv('./data/College.csv')
head(college)
rownames(college)= college[, 1]
college <- college[, -1]
head(college)
fix(college)

summary(college)
college[, 1:10]
windows()
pairs(college[, 1:10])
names(college)
boxplot(Outstate ~ Private, data=college, xlab='Private', ylab = 'Outstate')

head(college)
# Elite variable
Elite=rep("No", nrow(college))
Elite[college$Top10perc > 50] = "Yes" # boolean
Elite=as.factor(Elite)
college = data.frame(college, Elite)
summary(college)
boxplot(Outstate ~ Elite, data=college, xlab='Elite', ylab='Outstate')

par(mfrow=c(2, 2))
hist(college$Apps)
hist(college$Enroll)
hist(college$Expend)
hist(college$Outstate)














