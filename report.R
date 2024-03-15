
library(ggplot2)
library(dplyr)
library("data.table")
library("mlr3")
library("mlr3tuning")
library("mlr3learners")
library("mlr3verse")
library("glmnet")
require(KernSmooth)
library(MASS)
library("recipes")
library(pROC)
library(PRROC)

data <- read.csv("https://www.louisaslett.com/Courses/MISCADA/bank_personal_loan.csv")
data$Personal.Loan <- factor(data$Personal.Loan)
data[data$ZIP.Code == 9307, "ZIP.Code"] <- 94720
set.seed(123)


index <- sample(1:nrow(data), size = 0.8 * nrow(data))
training_test_data <- data[index, ] #data later split into folds for cross validation 
validation_data <- data[-index, ] #remaining data used to give final model accuaacies


loan_task <- TaskClassif$new(id = "Loan",backend = training_test_data, target = "Personal.Loan")
validate_task <- TaskClassif$new(id = "Loan",backend = validation_data, target = "Personal.Loan")


cv5 <- rsmp("cv", folds = 5) #set up folds for cross validation
cv5$instantiate(loan_task)

#BASELINE MODEL -------------------------------------------------------------------------
lrn_baseline <- lrn("classif.featureless", predict_type = "prob")
lrn_baseline$train(loan_task)
predictions_base <- lrn_baseline$predict(task = validate_task)
predictions_base$confusion
predictions_base$score(list(msr("classif.ce"),
                            msr("classif.acc"),
                            msr("classif.auc"),
                            msr("classif.fpr"),
                            msr("classif.fnr"),
                            msr("classif.tpr"),
                            msr("classif.tnr")))


#RANDOM FORREST -------------------------------------------------------------------------
tree_depths <- seq(1,15,by=1)
rfd_err <- list()
for(i in tree_depths){
  lrn_rf <- lrn("classif.ranger", predict_type = "prob", max.depth = i)
  res_rf <- resample(loan_task, lrn_rf, cv5, store_models = TRUE)
  rfd_err <- append(rfd_err, res_rf$aggregate())
}
plot(tree_depths,rfd_err, 
     main="Mean Misclassification Error Vs Tree Depth",
     ylab='Mean Misclassification Error',
     xlab='Tree Depth')
fit<- locpoly(tree_depths, rfd_err, bandwidth=1)
lines(fit, col=2)

trees <- seq(6,200, by=2)
rf_err <- list()
for(i in trees){
  lrn_rf <- lrn("classif.ranger", predict_type = "prob", max.depth = 10,num.trees = i)
  res_rf <- resample(loan_task, lrn_rf, cv5, store_models = TRUE)
  rf_err <- append(rf_err, res_rf$aggregate())
}
plot(trees,rf_err,main="Mean Misclassification Error Vs Number of Trees",
     ylab='Mean Misclassification Error',
     xlab='Number of Trees',
     ylim = c(0.0147, 0.019))
fit<- locpoly(trees, rf_err, bandwidth=14)
lines(fit, col=2)

lrn_rf <- lrn("classif.ranger", predict_type = "prob", max.depth = 10, num.trees = 100)
lrn_rf$train(loan_task)
predictions_rf <- lrn_rf$predict(task = validate_task)
predictions_rf$confusion
predictions_rf$score(list(msr("classif.ce"),
                          msr("classif.acc"),
                          msr("classif.auc"),
                          msr("classif.fpr"),
                          msr("classif.fnr"),
                          msr("classif.tpr"),
                          msr("classif.tnr")))


#GRADIENT BOOSTING ----------------------------------------------------------------------

nrounds <- seq(1,80,by=1)
nr_err <- numeric(length(nrounds))
for(i in nrounds){
  lrn_xbg <- lrn("classif.xgboost", predict_type = "prob",nrounds = i)
  res_xbg <- resample(loan_task, lrn_xbg, cv5, store_models = TRUE)
  nr_err[i] <- res_xbg$aggregate()
}

plot(nrounds, nr_err,main="Mean Misclassification Error Vs Number of GB Rounds",
     ylab='Mean Misclassification Error',
     xlab='Number of GB Rounds')
fit<- locpoly(nrounds, nr_err, bandwidth=1.6)
lines(fit, col=2)


depths <- seq(1,12,by=1)
d_err <- numeric(length(depths))
for(i in depths){
  lrn_xbg <- lrn("classif.xgboost", predict_type = "prob",nrounds = 25, max_depth = i)
  res_xbg <- resample(loan_task, lrn_xbg, cv5, store_models = TRUE)
  d_err[i] <- res_xbg$aggregate()
}
plot(depths,d_err,main="Mean Misclassification Error Vs Maximum Depth",
     ylab='Mean Misclassification Error',
     xlab='Maximum Depth')
fit<- locpoly(depths, d_err, bandwidth=0.8)
lines(fit, col=2)

lrn_xbg <- lrn("classif.xgboost", predict_type = "prob",nrounds = 25, max_depth = 10)
lrn_xbg$train(loan_task)
predictions_xgb <- lrn_xbg$predict(task = validate_task)
predictions_xgb$confusion
predictions_xgb$score(list(msr("classif.ce"),
                           msr("classif.acc"),
                           msr("classif.auc"),
                           msr("classif.fpr"),
                           msr("classif.fnr"),
                           msr("classif.tpr"),
                           msr("classif.tnr")))

#LOGISTIC REGRESSION -----------------------------------------------------------------------

#Commented code gives feature selection
#fit <- glm(Personal.Loan ~ ., family = binomial, data = training_test_data)
#selected_model <- stepAIC(fit, direction = "both")
#summary(selected_model)

selected_data <- training_test_data[, c('Personal.Loan','Income','Family','CCAvg','Education','Securities.Account', 'CD.Account','Online','CreditCard'), drop = FALSE]
selected_data_validate <- validation_data[, c('Personal.Loan','Income','Family','CCAvg','Education','Securities.Account', 'CD.Account','Online','CreditCard'), drop = FALSE]

lr_task <- TaskClassif$new(id = "Loans",backend = selected_data, target = "Personal.Loan")
lr_validate_task <- TaskClassif$new(id = "Loans",backend = selected_data_validate, target = "Personal.Loan")
lrn_lr <- lrn("classif.log_reg", predict_type = "prob")
lrn_lr$train(lr_task)
predictions_lr <- lrn_lr$predict(task = lr_validate_task)
predictions_lr$confusion
predictions_lr$score(list(msr("classif.ce"),
                          msr("classif.acc"),
                          msr("classif.auc"),
                          msr("classif.fpr"),
                          msr("classif.fnr"),
                          msr("classif.tpr"),
                          msr("classif.tnr")))

#SINGLE LAYER NEURAL NETWORK ---------------------------------------------------------------

numeric_data <- NN_data[, c("Age", "Experience","Income","Mortgage", "CCAvg","Personal.Loan")]
categoric_data <- NN_data[,c("Family", "Education", "Personal.Loan", "Securities.Account", "CD.Account", "Online", "CreditCard")]

categoric_data$Family <- factor(categoric_data$Family)
categoric_data$Education <- factor(categoric_data$Education)
categoric_data$Securities.Account <- factor(categoric_data$Securities.Account)
categoric_data$CD.Account <- factor(categoric_data$CD.Account)
categoric_data$Online <- factor(categoric_data$Online)
categoric_data$CreditCard <- factor(categoric_data$CreditCard)
categoric_data$Personal.Loan <- as.integer(categoric_data$Personal.Loan)

NN_training_test_data_numeric <- numeric_data[index, ]
NN_validation_data_numeric <- numeric_data[-index, ]
NN_training_test_data_categorical <- categoric_data[index, ]
NN_validation_data_categorical <- categoric_data[-index, ]

cake_numerical <- recipe(Personal.Loan ~ ., data = NN_training_test_data_numeric) %>%
  step_center(all_numeric()) %>%
  step_scale(all_numeric()) %>%
  prep(training = NN_training_test_data_numeric)

cake_categorical <- recipe(Personal.Loan ~ ., data = NN_training_test_data_categorical) %>%
  step_dummy(all_nominal(), one_hot = TRUE) %>%
  prep(training = NN_training_test_data_categorical)

NN_train_final_numerical <- bake(cake_numerical, new_data = NN_training_test_data_numeric)
NN_validate_final_numerical <- bake(cake_numerical, new_data = NN_validation_data_numeric)
NN_train_final_categorical <- bake(cake_categorical, new_data = NN_training_test_data_categorical)
NN_validate_final_categorical <- bake(cake_categorical, new_data = NN_validation_data_categorical)

NN_train_final_categorical$Personal.Loan <- NULL
NN_validate_final_categorical$Personal.Loan <- NULL

NN_train_final <- cbind(NN_train_final_numerical, NN_train_final_categorical)
NN_validate_final <- cbind(NN_validate_final_numerical, NN_validate_final_categorical)

nn_task <- TaskClassif$new(id = "Loan",backend = NN_train_final, target = "Personal.Loan")
nn_test_task <- TaskClassif$new(id = "Loan",backend = NN_validate_final, target = "Personal.Loan")
cv5$instantiate(nn_task)

nn_sizes <- seq(1,65,by=1)
nn_err <- list()
for(i in nn_sizes){
  print(i)
  lrn_nn <- lrn("classif.nnet", predict_type = "prob", maxit=1000,MaxNWts = 250000, size = i)
  res_nn <- resample(nn_task, lrn_nn, cv5, store_models = FALSE)
  nn_err <- append(nn_err, res_nn$aggregate())
}

plot(nn_sizes, nn_err,
     ylim=c(0.01,0.072),main="Mean Misclassification Error Vs Hidden Layer Size",
     ylab='Mean Misclassification Error',
     xlab='Hidden Layer Size')
fit<- locpoly(nn_sizes, nn_err, bandwidth=3)
lines(fit, col=2)

lrn_nn <- lrn("classif.nnet", predict_type = "prob",MaxNWts = 25000, size = 50)
lrn_nn$train(nn_task)
predictions_nn <- lrn_nn$predict(task = nn_test_task)
predictions_nn$confusion
predictions_nn$score(list(msr("classif.ce"),
                          msr("classif.acc"),
                          msr("classif.auc"),
                          msr("classif.fpr"),
                          msr("classif.fnr"),
                          msr("classif.tpr"),
                          msr("classif.tnr")))

probs_nn <- predictions_nn$prob[,2]
probs_rf <- predictions_rf$prob[,2]
probs_xgb <- predictions_xgb$prob[,2]
probs_lr <- predictions_lr$prob[,2]
probs_base <- predictions_base$prob[,2]

roc_curve_nn <- roc(validation_data$Personal.Loan, probs_nn)
roc_curve_rf <- roc(validation_data$Personal.Loan, probs_rf)
roc_curve_xgb <- roc(validation_data$Personal.Loan, probs_xgb)
roc_curve_lr <- roc(validation_data$Personal.Loan, probs_lr)
roc_curve_base <- roc(validation_data$Personal.Loan, probs_base)

roc_nn <- coords(roc_curve_nn)
x_nn <- roc_nn$specificity
y_nn <- roc_nn$sensitivity

roc_rf <- coords(roc_curve_rf)
x_rf <- roc_rf$specificity
y_rf <- roc_rf$sensitivity

roc_xgb <- coords(roc_curve_xgb)
x_xgb <- roc_xgb$specificity
y_xgb <- roc_xgb$sensitivity

roc_lr <- coords(roc_curve_lr)
x_lr <- roc_lr$specificity
y_lr <- roc_lr$sensitivity

roc_base <- coords(roc_curve_base)
x_base <- roc_base$specificity
y_base <- roc_base$sensitivity


#pdf("ROC.pdf", width = 6, height = 6)
plot(1-x_xgb,y_xgb, 
     main = "ROC Curves For Final Models",
     xlab="FPR",
     ylab="TPR",
     col = "green",
     type = "l",
     xlim =c(0,1),
     ylim = c(0,1))
lines(1-x_base,y_base,col='grey')
lines(1-x_rf,y_rf,col='red')
lines(1-x_nn,y_nn,col='orange')
lines(1-x_lr,y_lr, col = 'blue')
legend("bottomright", legend = c("Gradient Boosting","Random Forest", "Neural Network","Logistic Regression", "Baseline"), col = c("green","red","orange","blue", "grey"), lwd = 2)
#dev.off()

weights <- as.numeric(as.character(validation_data$Personal.Loan))

#pr_curve_base <- pr.curve(scores.class0 = probs_base, weights.class0 = weights, curve=T)
pr_curve_lr <- pr.curve(scores.class0 = probs_lr, weights.class0 = weights, curve=T)
pr_curve_rf <- pr.curve(scores.class0 = probs_rf, weights.class0 = weights, curve=T)
pr_curve_xgb <- pr.curve(scores.class0 = probs_xgb, weights.class0 = weights, curve=T)
pr_curve_nn <- pr.curve(scores.class0 = probs_nn, weights.class0 = weights, curve=T)

#pdf("PR.pdf", width = 6, height = 6)
plot(pr_curve_lr$curve[,1],pr_curve_lr$curve[,2], 
     main = "Precision-Recall Curve For Final Models", 
     col = "blue", type='l',
     xlab = "Recall",
     ylab = "Precision",
     ylim = c(0,1))

lines(c(0,1),c(0.095,0.095),col='grey')
lines(pr_curve_xgb$curve[,1],pr_curve_xgb$curve[,2], col = "green")
lines(pr_curve_rf$curve[,1],pr_curve_rf$curve[,2], col = "red")
lines(pr_curve_nn$curve[,1],pr_curve_nn$curve[,2], col = "orange")
lines(pr_curve_lr$curve[,1],pr_curve_lr$curve[,2], col = "blue")
legend("bottomleft", legend = c("Gradient Boosting", "Random Forest","Neural Network" ,"Logistic Regression","Baseline"), col = c("green", "red","orange", "blue","grey"), lwd = 2)
#dev.off()

predictions_xgb$confusion
predictions_xgb$score(list(msr("classif.ce"),
                           msr("classif.acc"),
                           msr("classif.auc"),
                           msr("classif.fpr"),
                           msr("classif.fnr"),
                           msr("classif.tpr"),
                           msr("classif.tnr")))
predictions_rf$confusion
predictions_rf$score(list(msr("classif.ce"),
                          msr("classif.acc"),
                          msr("classif.auc"),
                          msr("classif.fpr"),
                          msr("classif.fnr"),
                          msr("classif.tpr"),
                          msr("classif.tnr")))
predictions_nn$confusion
predictions_nn$score(list(msr("classif.ce"),
                          msr("classif.acc"),
                          msr("classif.auc"),
                          msr("classif.fpr"),
                          msr("classif.fnr"),
                          msr("classif.tpr"),
                          msr("classif.tnr")))
predictions_lr$confusion
predictions_lr$score(list(msr("classif.ce"),
                          msr("classif.acc"),
                          msr("classif.auc"),
                          msr("classif.fpr"),
                          msr("classif.fnr"),
                          msr("classif.tpr"),
                          msr("classif.tnr")))
predictions_base$confusion
predictions_base$score(list(msr("classif.ce"),
                            msr("classif.acc"),
                            msr("classif.auc"),
                            msr("classif.fpr"),
                            msr("classif.fnr"),
                            msr("classif.tpr"),
                            msr("classif.tnr")))

library(yardstick)
library(ggplot2)


truth_predicted<-data.frame(
  obs = validation_data$Personal.Loan,
  pred_lr = predictions_lr$response,
  pred_rf = predictions_rf$response,
  pred_xgb = predictions_xgb$response,
  pred_nn = predictions_nn$response
)

#pdf("confmat.pdf", width = 6, height = 6)
cm_rf <- conf_mat(truth_predicted, obs, pred_rf)
autoplot(cm_rf, type = "heatmap") +
  scale_fill_gradient(low="#D6EAF8",high = "#2E86C1")+
  ggtitle("Confusion Matrix - Random Forest")
#dev.off()

num_bins <- 12
bin_boundaries <- seq(0, 1, length.out = num_bins + 1)
bin_mid <- (bin_boundaries[-1] + bin_boundaries[-length(bin_boundaries)]) / 2

bins_lr <- cut(probs_lr, breaks = bin_boundaries)
proportion_of_1s_lr <- tapply(validation_data$Personal.Loan, bins_lr, function(x) mean(x == 1))

bins_rf <- cut(probs_rf, breaks = bin_boundaries)
proportion_of_1s_rf <- tapply(validation_data$Personal.Loan, bins_rf, function(x) mean(x == 1))

rf_x <- list()
rf_y <- list()
for(i in 1:length(proportion_of_1s_rf)){
  if(!is.na(proportion_of_1s_rf[i])){
    
    rf_y <- append(rf_y, proportion_of_1s_rf[i])
    rf_x <- append(rf_x, bin_mid[i])
  }
}

bins_xgb <- cut(probs_xgb, breaks = bin_boundaries)
proportion_of_1s_xgb <- tapply(validation_data$Personal.Loan, bins_xgb, function(x) mean(x == 1))

xgb_x <- list()
xgb_y <- list()
for(i in 1:length(proportion_of_1s_xgb)){
  if(!is.na(proportion_of_1s_xgb[i])){
    
    xgb_y <- append(xgb_y, proportion_of_1s_xgb[i])
    xgb_x <- append(xgb_x, bin_mid[i])
  }
}

bins_nn <- cut(probs_nn, breaks = bin_boundaries)
proportion_of_1s_nn <- tapply(validation_data$Personal.Loan, bins_nn, function(x) mean(x == 1))

nn_x <- list()
nn_y <- list()
for(i in 1:length(proportion_of_1s_nn)){
  if(!is.na(proportion_of_1s_nn[i])){
    
    nn_y <- append(nn_y, proportion_of_1s_nn[i])
    nn_x <- append(nn_x, bin_mid[i])
  }
}
#pdf("CC.pdf", width = 6, height = 6)
plot(c(0,1),c(0,1),col="grey",ylim=c(0,1),xlim=c(0,1),type='l',
     main="Calibration Curve For Final Models",
     ylab="Observed Event Fraction",
     xlab="Bin Midpoint")
lines(bin_mid, proportion_of_1s_lr,col='blue',lty = 1, lwd = 1, pch = 16,type="o")
lines(rf_x, rf_y,col='red',lty = 1, lwd = 1, pch = 16,type="o")
lines(xgb_x, xgb_y,col='green',lty = 1, lwd = 1, pch = 16,type="o")
lines(nn_x, nn_y,col='orange',lty = 1, lwd = 1, pch = 16,type="o")
legend("topleft", legend = c("Gradient Boosting", "Random Forest","Neural Network" ,"Logistic Regression"), col = c("green", "red","orange", "blue"), lwd = 2)
#dev.off()

lr_res <- bin_mid - proportion_of_1s_lr
least_squares_lr <- sum(lr_res^2) / length(lr_res)
least_squares_lr
rf_res <- unlist(rf_x) - unlist(rf_y)
least_squares_rf <- sum(rf_res^2) / length(rf_res)
least_squares_rf
xgb_res <- unlist(xgb_x) - unlist(xgb_y)
least_squares_xgb <- sum(xgb_res^2) / length(xgb_res)
least_squares_xgb
nn_res <- unlist(nn_x) - unlist(nn_y)
least_squares_nn <- sum(nn_res^2) / length(nn_res)
least_squares_nn

hist(data$Age, breaks= 17, main = "Histogram of Age Observations",xlab='Age (Years)',col = "paleturquoise")
hist(data$CCAvg, breaks= 13, main = "Histogram of CCAvg Observations",xlab='CCAvg ($1000s)',col = "paleturquoise")
hist(data$Experience, breaks= 17, main = "Histogram of Experience Observations",xlab='Expereince (Years)',col = "paleturquoise")
hist(data$Income, breaks= 16, main = "Histogram of Income Observations",xlab='Income ($1000s)',col = "paleturquoise")
non_zero_morgages <- unlist(lapply(data$Mortgage, function(x) subset(x, x != 0)))
corresponding_personal_loan <- data$Personal.Loan[data$Mortgage != 0]
zero_mort_personal_loan <- data$Personal.Loan[data$Mortgage == 0]
mort_data <- data.frame(corresponding_personal_loan, non_zero_morgages)
hist(non_zero_morgages, breaks= 15, main = "Histogram of Mortgage Observations\n(Only Non Zero Values Shown)",xlab='Mortgage Value ($1000s)',col = "paleturquoise")
barplot(table(data$Personal.Loan), main = "Bar Chart of Personal Loan Acceptance", ylab = "Frequency", names.arg = c("Rejected", "Accepted"), col = "peachpuff")
barplot(table(data$Education), main = "Bar Chart of Education Level", ylab = "Frequency",names.arg =c("Undergraduate", "Graduate", "Advanced/Professional"), col = "peachpuff")
barplot(table(data$Family), main = "Bar Chart of Family Size", ylab = "Frequency", col = "peachpuff")
barplot(table(data$CreditCard), main = "Bar Chart of if Customer Uses Credit Card", ylab = "Frequency", names.arg = c("No", "Yes"), col = "peachpuff")
barplot(table(data$Online), main = "Bar Chart of if Customer is Online", ylab = "Frequency", names.arg = c("No", "Yes"), col = "peachpuff")
barplot(table(data$CD.Account), main = "Bar Chart of if Customer Has A Certificate of Deposit", ylab = "Frequency", names.arg = c("No", "Yes"), col = "peachpuff")
barplot(table(data$Securities.Account), main = "Bar Chart of if Customer Has A Securities Account", ylab = "Frequency", names.arg = c("No", "Yes"), col = "peachpuff")

boxplot(Income ~ Personal.Loan, data = data, main = "Box Plot of Target Vs Income",xlab = "Income ($1000s)",ylab = "Loan Decison (Yes=1, No=2)",
        col = c("skyblue", "lightgreen"),names = c("0", "1"),border = "black",horizontal = TRUE) 

boxplot(Age ~ Personal.Loan, data = data, main = "Box Plot of Target Vs Age",xlab = "Age (Years)",ylab = "Loan Decison (Yes=1, No=2)",
        col = c("skyblue", "lightgreen"),names = c("0", "1"),border = "black",horizontal = TRUE) 

boxplot(CCAvg ~ Personal.Loan, data = data, main = "Box Plot of Target Vs CCAvg",xlab = "Average Credit Card Spend per Month ($1000s)",ylab = "Loan Decison (Yes=1, No=2)",
        col = c("skyblue", "lightgreen"),names = c("0", "1"),border = "black",horizontal = TRUE) 

boxplot(Experience ~ Personal.Loan, data = data, main = "Box Plot of Target Vs Experience",xlab = "Income ($1000s)",ylab = "Loan Decison (Yes=1, No=2)",col = c("skyblue", "lightgreen"),names = c("0", "1"),border = "black",horizontal = TRUE)

boxplot(non_zero_morgages ~ corresponding_personal_loan, data = mort_data, main = "Box Plot of Target Vs Non-Zero Mortgages",xlab = "Mortgage Value ($1000s)",ylab = "Loan Decison (Yes=1, No=2)",col = c("skyblue", "lightgreen"),names = c("0", "1"),border = "black",horizontal = TRUE)

frequency <- table(data$Personal.Loan, data$Family)
f1p <- frequency[2]/(frequency[1] + frequency[2])
f2p <- frequency[4]/(frequency[3] + frequency[4])
f3p <- frequency[6]/(frequency[5] + frequency[6])
f4p <- frequency[8]/(frequency[7] + frequency[8])
values <- 100 * c(f1p, f2p, f3p, f4p)
barplot(values, 
        main = "Bar Chart of Personal Loan Acceptance\nVs Family Size",
        ylab = "Personal Loan Acceptance (%)",
        names.arg =c("1", "2", "3", "4"),
        col = "skyblue",
        ylim = c(0,15))
abline(h = 100*48/500, col = "red", lty = 2, lwd = 2)
legend("topleft", legend = "Average (9.6%)", col = "red", lty = 2, lwd = 2,bty = "n")

frequency <- table(data$Personal.Loan, data$Education)
e1p <- frequency[2]/(frequency[1] + frequency[2])
e2p <- frequency[4]/(frequency[3] + frequency[4])
e3p <- frequency[6]/(frequency[5] + frequency[6])
values <- 100 * c(e1p, e2p, e3p)
barplot(values, 
        main = "Bar Chart of Personal Loan Acceptance\nVs Education",
        ylab = "Personal Loan Acceptance (%)",
        names.arg =c("Undergraduate", "Graduate", "Advanced/Professional"),
        col = "skyblue",
        ylim = c(0,15))
abline(h = 100*48/500, col = "red", lty = 2, lwd = 2)
legend("topleft", legend = "Average (9.6%)", col = "red", lty = 2, lwd = 2,bty = "n")


frequency <- table(data$Personal.Loan, data$CD.Account)
cd0p <- frequency[2]/(frequency[1] + frequency[2])
cd1p <- frequency[4]/(frequency[3] + frequency[4])
values <- 100 * c(cd0p, cd1p)
barplot(values, 
        main = "Bar Chart of Personal Loan Acceptance\nDepending on Having a COD",
        ylab = "Personal Loan Acceptance (%)",
        names.arg = c("No COD", "Has COD"),
        col = "skyblue")
abline(h = 100*48/500, col = "red", lty = 2, lwd = 2)
legend("topleft", legend = "Average (9.6%)", col = "red", lty = 2, lwd = 2,bty = "n")

