get.phat.nn <- function(Xtrain, train.df, Xtest, test.df){
    
    # building the model (going to use boosting)
    hyper_grid <- expand.grid(
        layer_1_units = c(32, 16),
        layer_2_units = c(16, 8),
        learning_rate = c(0.01, 0.1, 0.2),
        max_accuracy = 0,
        loss = 0
    )
    
    phat.nn.matrix <- matrix(0, nrow(test.df), nrow(hyper_grid))
    
    # grid search
    for(i in 1:nrow(hyper_grid)) {
        
        model <- keras_model_sequential()
        
        model %>%
            layer_dense(units = hyper_grid$layer_1_units[i], activation = 'relu', input_shape = c(52)) %>%
            layer_dense(units = hyper_grid$layer_2_units[i], activation = 'relu') %>%
            layer_dense(units = 2, activation = 'softmax')
        
        optimizer <- optimizer_sgd(
            learning_rate = hyper_grid$learning_rate[i],
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
        
        
        Xtrain_nn <- Xtrain
        Xtrain_nn <- apply(Xtrain_nn, MARGIN = 2, FUN = function(x){(x-min(x))/(max(x)-min(x))})
        dimnames(Xtrain_nn) = NULL
        Xtest_nn <- Xtest
        Xtest_nn <- apply(Xtest_nn, MARGIN = 2, FUN = function(x){(x-min(x))/(max(x)-min(x))})
        dimnames(Xtest_nn) = NULL

        Ytrain <- as.numeric(train.df$y) - 1
        Ytrain <- to_categorical(Ytrain)
        Ytest <- as.numeric(test.df$y) - 1
        Ytest <- to_categorical(Ytest)
        
        history <- model %>% 
            fit(Xtrain_nn, Ytrain, epochs = EPOCHS,
                validation_data = list(Xtest_nn, Ytest), verbose = 0)
        
        score <- model %>% evaluate(Xtest_nn, Ytest, verbose = 0)
        
        hyper_grid$max_accuracy[i] <- score['accuracy']
        hyper_grid$loss[i] <- score['loss']
        
        phat <- predict(model, Xtest_nn)[, 2]
        phat.nn.matrix[,i] <- phat
    }
    
    j.best <- which.min(hyper_grid$loss)
    phat.best <- phat.nn.matrix[,j.best]
    
    # sort
    oo = hyper_grid %>%
        dplyr::arrange(loss)
    
    return(list(phat.best, oo[1,]))
    
}