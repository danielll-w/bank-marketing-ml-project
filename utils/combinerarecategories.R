combinerarecategories<-function(data_frame,mincount){
    for (i in 1:ncol(data_frame)) {
        a<-data_frame[,i]
        replace <- names(which(table(a) < mincount))
        levels(a)[levels(a) %in% replace] <-
            paste("Other", colnames(data_frame)[i], sep=".")
        data_frame[,i]<-a
    }
    return(data_frame)
} 