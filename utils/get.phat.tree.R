get.phat.tree <- function(df.train, df.test){
    
    hyper_grid_tree <- expand.grid(
        minsplit = c(2, 3, 5, 10),
        cp = c(0.0001, 0.00001),
        xval = c(5, 10, 20)
    )
    
    phat.tree.matrix <- matrix(0, nrow(df.test), nrow(hyper_grid_tree))
    
    for(j in 1:nrow(hyper_grid_tree)){
        big.tree <- rpart(y ~ ., data=df.train,
                          control=rpart.control(
                              minsplit=hyper_grid_tree$minsplit[j],
                              cp=hyper_grid_tree$cp[j],
                              xval=hyper_grid_tree$xval[j]
                          ))
        cptable <- printcp(big.tree)
        bestcp <- cptable[which.min(cptable[,"xerror"]), "CP"]
        best.tree <- prune(big.tree, cp=bestcp)
        phat <- predict(best.tree, df.test)[, 2]
        phat.tree.matrix[,j] <- phat
    }
    
    auc <- rep(0, nrow(hyper_grid_tree))
    for(j in 1:nrow(hyper_grid_tree)){
        # AUC
        pred = prediction(phat.tree.matrix[,j], df.test$y)
        perf = performance(pred, measure = "aucpr")
        auc[j] <- perf@y.values[[1]]
    }
    j.best <- which.max(auc)
    phat.best <- phat.tree.matrix[,j.best]
    
    # sort by auc
    (oo = hyper_grid_tree %>%
            mutate(auc = auc) %>% 
            dplyr::arrange(desc(auc)) %>%
            head(10))
    
    # best parameters
    return(list(phat.best, max(auc), oo[1,]))

}

