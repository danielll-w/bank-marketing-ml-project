# Create a custom function to fix missing values ("NAs") and
# preserve the NA info as surrogate variables
fixNAs <- function(data_frame){
    # Define reactions to NAs
    integer_reac <- 0
    factor_reac <- "FIXED_NA"
    character_reac <- "FIXED_NA"
    date_reac <- as.Date("1900-01-01")
    # Loop through columns in the data frame
    # and depending on which class the
    # variable is, apply the defined reaction and
    # create a surrogate
    for (i in 1:ncol(data_frame)) {
        if (class(data_frame[,i]) %in% c("numeric","integer")) {
            if (any(is.na(data_frame[,i]))) {
                data_frame[,paste0(colnames(data_frame)[i],"_surrogate")] <-
                    as.factor(ifelse(is.na(data_frame[,i]),"1","0"))
                data_frame[is.na(data_frame[,i]), i] <- integer_reac
            }
        } else
            if (class(data_frame[,i]) %in% c("factor")) {
                if (any(is.na(data_frame[,i]))){
                    data_frame[,i]<-as.character(data_frame[,i])
                    data_frame[,paste0(colnames(data_frame)[i],"_surrogate")] <-
                        as.factor(ifelse(is.na(data_frame[,i]),"1","0"))
                    data_frame[is.na(data_frame[,i]),i]<-factor_reac
                    data_frame[,i]<-as.factor(data_frame[,i])
                }
            } else {
                if (class(data_frame[,i]) %in% c("character")) {
                    if (any(is.na(data_frame[,i]))){
                        data_frame[,paste0(colnames(data_frame)[i],"_surrogate")]<-
                            as.factor(ifelse(is.na(data_frame[,i]),"1","0"))
                        data_frame[is.na(data_frame[,i]),i]<-character_reac
                    }
                } else {
                    if (class(data_frame[,i]) %in% c("Date")) {
                        if (any(is.na(data_frame[,i]))){
                            data_frame[,paste0(colnames(data_frame)[i],"_surrogate")]<-
                                as.factor(ifelse(is.na(data_frame[,i]),"1","0"))
                            data_frame[is.na(data_frame[,i]),i]<-date_reac
                        }
                    }
                }
            }
    }
    return(data_frame)
}