
# y should be 0/1
# phat are probabilities obtained by our algorithm
# wht shrinks probabilities in phat towards .5
# this helps avoid numerical problems --- don't use log(0)!
lossf = function(y,phat,wht=0.0000001) {
    if(is.factor(y)) y = as.numeric(y)-1
    phat = (1-wht)*phat + wht*.5
    py = ifelse(y==1, phat, 1-phat)
    return(-2*sum(log(py)))
}