get.phat.rf <- function(df.train, df.test){
    
    p <- ncol(df.train) - 1
    
    hyper_grid_rf <- expand.grid(
        mtry       = seq(2, ceiling(sqrt(p)), by = 1),
        node_size  = c(10, 25, 50),
        max.depth = c(1, 2, 3, 5, 10)
    )
    
    phat.rf.matrix <- matrix(0, nrow(df.test), nrow(hyper_grid_rf))
    
    for(j in 1:nrow(hyper_grid_rf)){
        # Train model
        rf.model <- ranger(
            formula = y ~ .,
            data=df.train,
            num.trees = 500,
            mtry = hyper_grid_rf$mtry[j],
            min.node.size = hyper_grid_rf$node_size[j],
            max.depth = hyper_grid_rf$max.depth[j],
            probability = T,
            seed = 233
        )
        phat <- predict(rf.model, data = df.test)$predictions[, 2]
        phat.rf.matrix[,j] <- phat
    }
    
    auc <- rep(0, nrow(hyper_grid_rf))
    for(j in 1:nrow(hyper_grid_rf)){
        # AUC
        pred = prediction(phat.rf.matrix[,j], df.test$y)
        perf = performance(pred, measure = "aucpr")
        auc[j] <- perf@y.values[[1]]
        
        # auc[j] <- lossf(test.df$y, phat.rf.matrix[,j], wht=0.0000001)
        
    }
    j.best <- which.min(auc)
    phat.best <- phat.rf.matrix[,j.best]
    
    # sort by auc
    (oo = hyper_grid_rf %>%
            mutate(auc = auc) %>% 
            dplyr::arrange(desc(auc)) %>%
            head(10))
    
    # best parameters
    return(list(phat.best, max(auc), oo[1,]))
    
}

