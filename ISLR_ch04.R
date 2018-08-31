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
























