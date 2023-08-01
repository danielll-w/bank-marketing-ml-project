get.phat.xgb <- function(df.train, df.test){
    
    # Hypergrid of XGBoost parameters
    hyper_grid_xgb <- expand.grid(
        shrinkage =c(.01, .1, .3), 
        interaction.depth =c(1, 2, 3, 5),
        n.minobsinnode = c(10, 30, 50),
        bag.fraction = c(.5, .65, .8)
    )
    
    set.seed(233)
    phat.xgb.matrix <- matrix(0.0, nrow(df.test), nrow(hyper_grid_xgb))
    
    for(j in 1:nrow(hyper_grid_xgb)){
        
        X.train <- df.train
        X.test <- df.test
        
        X.train =  sparse.model.matrix(y ~ ., data = X.train)[,-1]
        Y.train = as.matrix(as.numeric(df.train$y == "1"))
        
        X.test =  sparse.model.matrix(y ~ ., data = X.test)[,-1]
        Y.test = as.matrix(as.numeric(df.test$y == "1"))
        
        params <- list(
            eta = hyper_grid_xgb$shrinkage[j],
            max_depth = hyper_grid_xgb$interaction.depth[j],
            min_child_weight = hyper_grid_xgb$n.minobsinnode[j],
            subsample = hyper_grid_xgb$bag.fraction[j]
        )
        
        xgb.model <- xgboost(
            data = X.train,
            label = Y.train,
            params = params,
            nrounds = 1000,
            objective = "binary:logistic",
            verbose = 0,
            verbosity = 0
        )
        phat <- predict(xgb.model, X.test)
        phat.xgb.matrix[,j] <- phat
    }
    
    auc <- rep(0, nrow(hyper_grid_xgb))
    for(j in 1:nrow(hyper_grid_xgb)){
        # AUC
        pred = prediction(phat.xgb.matrix[,j], df.test$y)
        perf = performance(pred, measure = "aucpr")
        auc[j] <- perf@y.values[[1]]
    }
    j.best <- which.max(auc)
    phat.best <- phat.xgb.matrix[,j.best]
    
    # sort by auc
    (oo = hyper_grid_xgb %>%
            mutate(auc = auc) %>% 
            dplyr::arrange(auc) %>%
            head(10))
    
    # best parameters
    return(list(phat.best, max(auc), oo[1,]))
    
    # validation_error <- rep(0, nrow(hyper_grid_xgb))
    # for(j in 1:nrow(hyper_grid_xgb)){
    #     phat <- phat.xgb.matrix[,j]
    #     p_class <- phat > 0.5
    #     validation_error[j] <- 1 - (sum(as.numeric(p_class) == as.numeric(df.test$y) - 1) / nrow(df.test))
    # }
    # j.best <- which.min(validation_error)
    # phat.best <- phat.xgb.matrix[, j.best]
    # 
    # # sort by error
    # (oo = hyper_grid_xgb %>%
    #         dplyr::arrange(validation_error) %>%
    #         head(10))
    # 
    # # best parameters
    # return(list(phat.best, min(validation_error), oo[1,]))

}
