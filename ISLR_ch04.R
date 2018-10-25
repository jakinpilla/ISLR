# 4. classification

# 4.1 An overview of classification
# Default dataset, 

# 4.2 Why not linear regression?

# 4.3 Logistic Regession
# Logistic regression models calculate the probability that Y belongs to a particular category
# Pr(default=Yes|balance)

# 4.3.1 The Logistic Model
# logistic function
# odds

# 4.3.2 Estimating the Regression Coefficients
# the more general method of maximum likelihood is prefered since it has better statistical 
# properties
# likelihood function

# 4.3.3 Making Predictions

# 4.3.4 Multiple Logistic Regression
# confounding

# 4.3.5 Logistic Regression for > 2 Response Classes

# 4.4 Linear Discriminant Analysis
## When the classes are well-seperated, the parameter estimates for the 
## logistic regression model are surprisingly unstable. Linear discriminant
## analysis does not suffer from this problem.

## If n is small and the distribution of the predictors X is approximately
## normal in each of the classes, the linear discreminant model is again
# more stable than the logistic regression model.

## Linear discriminant analysis is popular when we have more than two response classes

# 4.4.1 Using Bayes' Theorem for Classification
# f_k(x) = Pr(X=x|Y=k) :: the density function of X for an observation that comes from
# the kth class. In other words, f_k(x) is relatively large if there is a high probability
# that an observation in the kth class has X = x, and f_k(x) is small if it is very unlikely
# that an observation in the kth class has X = x.

# we refer to p_k(x) as the posterior probability that an observation X=x belong to the kth class
# that is, it is the probability that the observation belongs to the kth class, given the predictor
# value for that observation

# 4.4.2 Linear Discriminant Analysis for p=1
# p=1 :: only one predictor
# pi_k :: the prior probability that observation belongs to the
# kth class

# 4.4.3 Linear Discriminant Analysis for p > 1
# ROC :: receiver operating characteristics

# 4.4.4 Quadratic Discriminant Analysis

# 4.5 A Comparison of Classification Methods

# 4.6 :: Lab : Logistic Regression, LDA, QDA, and KNN

library(ISLR)
names(Smarket)
dim(Smarket)
summary(Smarket)

cor(Smarket)
cor(Smarket[, -9])

attach(Smarket)
plot(Volume)

# 4.6.2 Logistic Regression
glm.fit= glm(Direction ~ Lag1 + Lag2 + Lag3 + Lag4 +
               Lag5 + Volume, family = binomial)
summary(glm.fit)

coef(glm.fit)
summary(glm.fit)$coef
summary(glm.fit)$coef[,4]

glm.probs=predict(glm.fit, type='response')
glm.probs[1:10]
contrasts(Direction)

glm.pred=rep('Down', 1250)
glm.pred[glm.probs > .5] = 'Up'

table(glm.pred, Direction)
(507+145)/1250
mean(glm.pred==Direction)

train=(Year<2005)
Smarket.2005 = Smarket[!train,]
dim(Smarket.2005)
Direction.2005 = Direction[!train]

glm.fit = glm(Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume,
              data=Smarket, family=binomial, subset=train)
glm.probs = predict(glm.fit, Smarket.2005, type='response')

glm.pred = rep('Down', 252)
glm.pred[glm.probs > .5]="Up"
table(glm.pred, Direction.2005)
mean(glm.pred==Direction.2005)
mean(glm.pred != Direction.2005)

glm.fit = glm(Direction ~ Lag1 + Lag2, data=Smarket, 
              family='binomial', subset=train)
glm.probs=predict(glm.fit, Smarket.2005, type='response')
glm.pred=rep('Down', 252)
glm.pred[glm.probs >.5]='Up'
table(glm.pred, Direction.2005)
mean(glm.pred==Direction.2005)

predict(glm.fit, newdata = data.frame(Lag1=c(1.2, 1.5),
                                      Lag2=c(1.1, -0.8)),
        type='response')

library(MASS)
lda.fit = lda(Direction ~ Lag1 + Lag2, data=Smarket, 
              subset=train)

lda.fit
lda.pred=predict(lda.fit, Smarket.2005)
names(lda.pred)

lda.pred = predict(lda.fit, Smarket.2005)
names(lda.pred)

lda.class = lda.pred$class
table(lda.class, Direction.2005)
mean(lda.class==Direction.2005)

sum(lda.pred$posterior[,1] >= .5)
sum(lda.pred$posterior[,1] < .5)
lda.pred$posterior[1:20, 1]
lda.class[1:20]

sum(lda.pred$posterior[,1] > .9)

# 4.6.4 Quadratic Discriminant Analysis

library(MASS)
qda.fit = qda(Direction ~ Lag1 + Lag2, data=Smarket, subset=train)
qda.fit

q = predict(qda.fit, Smarket.2005)$class
table(qda.class, Direction.2005)
mean(qda.class==Direction.2005)

library(class)

Lag1
Lag2
train.X = cbind(Lag1, Lag2)[train,]
test.X = cbind(Lag1, Lag2)[!train,]
train.Direction = Direction[train]
knn.pred = knn(train.X, test.X, train.Direction, k=1)
table(knn.pred, Direction.2005)
(83+43)/252

knn.pred = knn(train.X, test.X, train.Direction, k=3)
table(knn.pred, Direction.2005)
mean(knn.pred == Direction.2005)

# 4.6.6 An Application to Caravan Insurance Data
dim(Caravan)
head(Caravan)
attach(Caravan)
summary(Purchase)
348/5822
standardized.X = scale(Caravan[,-86])
var(Caravan[,1])
var(Caravan[,2])
var(standardized.X[, 1])
var(standardzied.X[, 2])

test=1:1000
train.X = standardized.X[-test, ]
test.X = standardized.X[test, ]
train.Y = Purchase[-test]
test.Y =  Purchase[test]
set.seed(1)
knn.pred = knn(train.X, test.X, train.Y, k=1)
mean(test.Y != knn.pred)
mean(test.Y != "No")

table(knn.pred, test.Y)

knn.pred=knn(train.X, test.X, train.Y, k=3)
table(knn.pred, test.Y)
5/26

knn.pred = knn(train.X, test.X, train.Y, k=5)
table(knn.pred, test.Y)
4/15

glm.fit = glm(Purchase ~. , data=Caravan, family = binomial, 
              subset=-test)
glm.probs = predict(glm.fit, Caravan[test, ], type='response')

glm.pred = rep('No', 1000)
glm.pred[glm.probs >.5]='Yes'
table(glm.pred, test.Y)

glm.pred = rep('No', 1000)
glm.pred[glm.probs > .25]='Yes'
table(glm.pred, test.Y)
11/(22+11)








