# Final Project
# Author: Dan Weiss

library(tidyverse)
library(fastDummies)
library(glmnet)
library(caret)
library(ranger)
library(xgboost)
library(keras)
library(tensorflow)
library(ROCR)
library(rpart)
library(rpart.plot)
library(gglasso)
library(Hmisc)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# source("fixNAs.R")
# source("combinerarecategories.R")
# source("getConfusionMatrix.R")
# source("lossf.R")
# source("get.phat.rf.R")
# source("get.phat.xgb.R")
# source("get.phat.nn.R")

# Load ------------------------------------------------------------------------------------------------------------


bank_data <- read_delim("raw-data/bank-additional-full.csv", delim = ";")


# Cleaning --------------------------------------------------------------------------------------------------------

str(bank_data)

# convert dependent variable to numeric
bank_data$y <- as.factor(bank_data$y)
levels(bank_data$y) <- c(0, 1)

# transform age
bank_data$age <- log(bank_data$age)

# combine categories in default
bank_data <- bank_data %>% mutate(default = ifelse(default == "yes", "unknown", default))

# combine categories in previous
bank_data <- bank_data %>% mutate(previous = ifelse(previous %in% c("2", "3", "4", "5", "6", "7"), "2", previous))
bank_data$previous <- factor(bank_data$previous, ordered = TRUE, levels = c("0", "1", "2"))

# assume unknown marital status is single
bank_data <- bank_data %>% mutate(marital = ifelse(marital == "unknown", "single", marital))

# drop illiterate rows
idx <- bank_data$education == "illiterate"
bank_data <- bank_data[!idx, ]

# deal with campaign variable
bank_data$campaign[bank_data$campaign > 5] = 5
bank_data$campaign <- factor(bank_data$campaign, ordered = TRUE, levels = c("1", "2", "3", "4", "5"))

# convert character variables to factor
bank_data <- mutate_at(bank_data, vars(job, marital, education, default, housing, loan, contact, month, day_of_week,
                                       poutcome), as.factor)

# convert "previous" variable to factor
bank_data$previous <- as.factor(bank_data$previous)

# drop pdays and duration
bank_data <- bank_data %>% select(-pdays) %>% select(-duration)


# Exploratory Data Analysis ---------------------------------------------------------------------------------------

# Percent Yes/No in dependent variable
table(bank_data$y) / nrow(bank_data)
hist(bank_data$age)

# tables with proportions of term deposit by factor level for job and education
bank_data %>% group_by(job) %>% summarise(percent_term_deposit = sum(y == "1") / n(), n())
bank_data %>% group_by(education) %>% summarise(percent_term_deposit = sum(y == "1") / n(), n())

# Modeling --------------------------------------------------------------------------------------------------------

# stratified sampling based on binary dependent variable to create training and validation sets
set.seed(213)
in.train = createDataPartition(bank_data$y, p=0.85, list=FALSE)

train.df = bank_data[in.train,]
test.df = bank_data[-in.train,]

in.validate <- createDataPartition(train.df$y, p=(0.15/0.85), list=FALSE)
validate.df = train.df[in.validate,]
train.df = train.df[-in.validate,]

Xtrain = model.matrix(y ~ ., train.df)[, -1]
Xvalidate = model.matrix(y ~ ., validate.df)[, -1]
Xtest = model.matrix(y ~ ., test.df)[, -1]

# lasso logistic regression
Xtrain_big <- rbind(Xtrain, Xvalidate)
Ytrain_big <- c(train.df$y, validate.df$y)

lassofit = cv.glmnet(x = Xtrain_big, y = Ytrain_big, family = "binomial", alpha = 1, nfolds = 10)
glmnet.fit = lassofit$glmnet.fit

phat_lasso = predict(glmnet.fit, newx = Xtest, s = lassofit$lambda.1se, type="response")
pred_lasso = prediction(phat_lasso, test.df$y)

plot(lassofit$glmnet.fit, xvar='lambda')

dev.copy(png,'lasso_lambda_plot.png')
dev.off()

lassofit$lambda.1se

# random forest
phat_rf_array <- get.phat.rf(train.df, validate.df)

params <- phat_rf_array[[3]]

Xtrain_full <- rbind(train.df, validate.df)
rf.fit = ranger(formula = y ~ ., data = Xtrain_full, num.trees=500, mtry=params$mtry[1], min.node.size=params$node_size[1], 
                max.depth = params$max.depth[1], probability=TRUE, seed = 41204, importance = 'impurity')

phat_rf = predict(rf.fit, test.df)$predictions[, 2]
pred_rf = prediction(phat_rf, test.df$y)

# importance plot
tvimp = sort(importance(rf.fit), decreasing = TRUE)
tvimp = tvimp[1:10]

# par(mar=c(8,5,1,1))
plot(tvimp/max(tvimp),axes=F,pch=16,col='red',xlab="",ylab="importance", main = "RF Importance", cex=2,cex.lab=1.5)
axis(1,labels=names(tvimp), at=1:length(tvimp), cex.axis=0.8, las=2)
axis(2)

dev.copy(png,'rf_var_imp.png')
dev.off()

phat_rf_array[[3]]

# boosting
phat_xgb_array <- get.phat.xgb(train.df, validate.df)

Xtrain_full <- rbind(train.df, validate.df)
Xtrain_full =  sparse.model.matrix(y ~ ., data = Xtrain_full)[,-1]

Ytrain_full <- c(train.df$y, validate.df$y)
Ytrain_full <- as.matrix(as.numeric(Ytrain_full == "1"))

params <- phat_xgb_array[[3]]
params <- list(eta = params$shrinkage[1], max_depth = params$interaction.depth[1], 
               min_child_weight = params$n.minobsinnode[1], subsample = params$bag.fraction)

xgb.model <- xgboost(
    data = Xtrain_full,
    label = Ytrain_full,
    params = params,
    nrounds = 1000,
    objective = "binary:logistic",
    verbose = 0,
    verbosity = 0
)

phat_xgb <- predict(xgb.model, sparse.model.matrix(y ~ ., data = test.df)[,-1])
pred_xgb = prediction(phat_xgb, test.df$y)

# create importance matrix
importance_matrix <- xgb.importance(model = xgb.model)

# variable importance plot
xgb.plot.importance(importance_matrix, measure = "Gain", top_n = 10)

dev.copy(png,'xgb_var_imp.png')
dev.off()

phat_xgb_array[[3]]

# neural net
phat_nn_array <- get.phat.nn(Xtrain, train.df, Xvalidate, validate.df)

params <- phat_nn[[2]]

model <- keras_model_sequential()

model %>%
    layer_dense(units = params$layer_1_units[1], activation = 'relu', input_shape = c(52)) %>%
    layer_dense(units = params$layer_2_units[1], activation = 'relu') %>%
    layer_dense(units = 2, activation = 'softmax')

optimizer <- optimizer_sgd(
    learning_rate = params$learning_rate[1],
    momentum = 0,
    decay = 0,
    nesterov = FALSE
)

model %>% compile(
    optimizer = optimizer,
    # Loss function
    loss = 'categorical_crossentropy',
    # Metric(s) used to evaluate the model while training
    metrics = c('accuracy')
)

# Set the number of times you want to go over the training dataset
EPOCHS <- 50

Xtrain_nn <- rbind(Xtrain, Xvalidate)
Xtrain_nn <- apply(Xtrain_nn, MARGIN = 2, FUN = function(x){(x-min(x))/(max(x)-min(x))})
dimnames(Xtrain_nn) = NULL
Xtest_nn <- Xtest
Xtest_nn <- apply(Xtest_nn, MARGIN = 2, FUN = function(x){(x-min(x))/(max(x)-min(x))})
dimnames(Xtest_nn) = NULL

Ytrain <- as.numeric(c(train.df$y, validate.df$y)) - 1
Ytrain <- to_categorical(Ytrain)
Ytest <- as.numeric(test.df$y) - 1
Ytest <- to_categorical(Ytest)

history <- model %>%
    fit(Xtrain_nn, Ytrain, epochs = EPOCHS,
        validation_data = list(Xtest_nn, Ytest), verbose = 0)

score <- model %>% evaluate(Xtest_nn, Ytest, verbose = 0)
cat('Test accuracy:', score['accuracy'], "\n")

# Get probabilities for each class per sample
phat_nn <- predict(model, Xtest_nn)[, 2]
pred_nn <- prediction(phat_nn, test.df$y)

phat_nn

# Model Metrics ---------------------------------------------------------------------------------------------------

par(mfrow=c(1,2))

# ROC
perf = performance(pred_lasso, measure = "tpr", x.measure = "fpr")
plot(perf, col = 1, lwd = 2,
     main= 'ROC curve', xlab='FPR', ylab='TPR', cex.lab=1)

perf = performance(pred_rf, measure = "tpr", x.measure = "fpr")
plot(perf, add = T, col = 2, lwd = 2)

perf = performance(pred_xgb, measure = "tpr", x.measure = "fpr")
plot(perf, add = T, col = 3, lwd = 2)

perf = performance(pred_nn, measure = "tpr", x.measure = "fpr")
plot(perf, add = T, col = 4, lwd = 2)

abline(0,1,lty=2)
legend("bottomright",legend=c('Lasso', 'RF', 'Boosting', 'NN'), col=1:4, lwd=2)

# ROC AUC
perf = performance(pred_lasso, measure = "auc")
auc <- perf@y.values[[1]]
auc

perf = performance(pred_rf, measure = "auc")
auc <- perf@y.values[[1]]
auc

perf = performance(pred_xgb, measure = "auc")
auc <- perf@y.values[[1]]
auc

perf = performance(pred_nn, measure = "auc")
auc <- perf@y.values[[1]]
auc

# PR Curve
perf = performance(pred_lasso, measure = "prec", x.measure = "rec")
plot(perf, col = 1, lwd = 2,
     main= 'Precision-Recall curve', xlab='Recall', ylab='Precision', cex.lab=1)

perf = performance(pred_rf, measure = "prec", x.measure = "rec")
plot(perf, add = T, col = 2, lwd = 2)

perf = performance(pred_xgb, measure = "prec", x.measure = "rec")
plot(perf, add = T, col = 3, lwd = 2)

perf = performance(pred_nn, measure = "prec", x.measure = "rec")
plot(perf, add = T, col = 4, lwd = 2)

abline(h=0.113, lty=2)

legend("topright",legend=c('Lasso', 'RF', 'Boosting', 'NN'), col=1:4, lwd=2)

# PR AUC
perf = performance(pred_lasso, measure = "aucpr")
auc <- perf@y.values[[1]]
auc

perf = performance(pred_rf, measure = "aucpr")
auc <- perf@y.values[[1]]
auc

perf = performance(pred_xgb, measure = "aucpr")
auc <- perf@y.values[[1]]
auc

perf = performance(pred_nn, measure = "aucpr")
auc <- perf@y.values[[1]]
auc

# LUC
par(mfrow=c(1,1))

perf = performance(pred_lasso, measure = "lift", x.measure = "rpp", lwd=2)
plot(perf, col=1, ylim=c(0.5, 10))

perf = performance(pred_rf, measure = "lift", x.measure = "rpp")
plot(perf, add = T, col = 2, lwd = 2)

perf = performance(pred_xgb, measure = "lift", x.measure = "rpp")
plot(perf, add = T, col = 3, lwd = 2)

perf = performance(pred_nn, measure = "lift", x.measure = "rpp")
plot(perf, add = T, col = 4, lwd = 2)

abline(h=1, lty=2)

legend("topright",legend=c('Lasso', 'RF', 'Boosting', 'NN'), col=1:4, lwd=2)

dev.copy(png,'Lift.png')
dev.off()


# Extra Model Metrics ---------------------------------------------------------------------------------------------

# plain confusion matrices
lasso_cf <- getConfusionMatrix(test.df$y, phat_lasso, 0.5)
rf_cf <- getConfusionMatrix(test.df$y, phat_rf, 0.5)
xgb_cf <- getConfusionMatrix(test.df$y, phat_xgb, 0.5)
nn_cf <- getConfusionMatrix(test.df$y, phat_nn, 0.5)

lasso_cf$byClass["Specificity"]
rf_cf$byClass["Specificity"]
xgb_cf$byClass["Specificity"]
nn_cf$byClass["Specificity"]

# deviance
lossf(test.df$y, phat_lasso)
lossf(test.df$y, phat_rf)
lossf(test.df$y, phat_xgb)
lossf(test.df$y, phat_nn)

# G-mean
sqrt(lasso_cf$byClass["Specificity"]*lasso_cf$byClass["Sensitivity"])
sqrt(rf_cf$byClass["Specificity"]*rf_cf$byClass["Sensitivity"])
sqrt(xgb_cf$byClass["Specificity"]*xgb_cf$byClass["Sensitivity"])
sqrt(nn_cf$byClass["Specificity"]*nn_cf$byClass["Sensitivity"])
