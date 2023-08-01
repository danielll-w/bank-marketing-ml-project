# y should be 0/1
# phat are probabilities obtained by our algorithm
# thr is the cut off value - everything above thr is classified as 1
getConfusionMatrix = function(y, phat, thr) {
    yhat = as.factor(ifelse(phat > thr, 1, 0))
    confusionMatrix(yhat, y)
}
