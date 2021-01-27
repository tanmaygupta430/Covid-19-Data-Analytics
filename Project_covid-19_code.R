library(tidyverse)
library(plyr)
library(tidyr)
library(rpart)
library(DMwR)
library(corrplot)
library(leaps)
library(UBL)
library(e1071)
library(class)
library(caret)
covid_USA <- read.csv('national-history.csv')
head(covid_USA)
#getting variable names
colnames(covid_USA)
#Renaming Variables
names(covid_USA[1])
colnames(covid_USA)[1]<- 'Date'
names(covid_USA[2])
colnames(covid_USA)[2]<- 'Total_death'
names(covid_USA[3])
colnames(covid_USA)[3]<- 'Today_death'
names(covid_USA[4])
colnames(covid_USA)[4]<- 'In_ICU_cummalative'
names(covid_USA[5])
colnames(covid_USA)[5]<- 'Total_in_ICU'
names(covid_USA[6])
colnames(covid_USA)[6]<- 'Today_hospitalized'
names(covid_USA[7])
colnames(covid_USA)[7]<- 'Total_hospitalized'
names(covid_USA[8])
colnames(covid_USA)[8]<- 'Hospitalized_cummalative'
names(covid_USA[9])
colnames(covid_USA)[9]<- 'Total_negative'
names(covid_USA[10])
colnames(covid_USA)[10]<- 'Today_negative'
names(covid_USA[11])
colnames(covid_USA)[11]<- 'Ventilator_cummalative' 
names(covid_USA[12])
colnames(covid_USA)[12]<- 'Total_on_ventilator'
names(covid_USA[13])
colnames(covid_USA)[13]<- 'Total_positive_cases'
names(covid_USA[14])
colnames(covid_USA)[14]<- 'Today_positive_cases'
names(covid_USA[15])
colnames(covid_USA)[15]<- 'Total_people_recovered'
names(covid_USA[17])
colnames(covid_USA)[17]<- 'Total_tests_results'
names(covid_USA[18])
colnames(covid_USA)[18]<- 'Total_tests_results_today'

#taking a look at the renamed data
glimpse(covid_USA) 

#removing date and states for anaysis
covid_USA_upd<- covid_USA[-c(1,16)]
#replacing na with 0
covid_USA_upd[is.na(covid_USA_upd)]<-0

#correlation between variables
round(cor(covid_USA_upd[,-1]),digits=2)

#plotting all the variables against our response variable 
covid_USA_upd %>%
  gather(key, val, -Today_positive_cases) %>%
  ggplot(aes(x = val, y = Today_positive_cases)) +
  geom_point() +
  stat_smooth(method = "lm", se = TRUE, col = "green") +
  facet_wrap(~key, scales = "free") +
  theme_gray() +
  ggtitle("Scatter plot of Dependent Variables vs Today_positive_cases")

#SmoteR for tackling imbalance 
smote_covid_2<- SmoteRegress(Today_positive_cases~., covid_USA_upd, C.perc = list(0.5,2.5)) 
#scaling data 
scaled_data<- scale(smote_covid_2)
#converting into a data frame
scaled_data_1<- as.data.frame(scaled_data)

#train-test split of the data 
set.seed(2020)
training_ind<- sample(nrow(scaled_data), nrow(scaled_data)* 0.8)
train<- scaled_data[training_ind, ]
train[is.na(train)]<- 0
train_1<- as.data.frame(train)
test<- scaled_data[-training_ind, ]
test_1<- as.data.frame(test)


#Linear Regression model
reg<- lm(Today_positive_cases~., data= train_1) 
info_reg<-summary(reg)

#predicting on test set
reg_test<- predict(reg, newdata = test_1)
head(reg_test)

#model with subset
model_1 <- regsubsets(Today_positive_cases ~., data= train_1, nbest=1, nvmax= 15)
summary(model_1)

#Variable selection using stepwise selection 
#Null linear model
nullmodel_1 <- lm(Today_positive_cases ~ 1, data = scaled_data_1)
#Full linear model
fullmodel_1 <- lm(Today_positive_cases ~., data = scaled_data_1)
#forward selection
model_forward<-step(nullmodel_1, scope = list(lower = nullmodel_1, upper = fullmodel_1), direction = "forward")
#backward model
model.step.b <- step(fullmodel_1, direction = "backward")

#stepwise selection
model_3_1<- step(nullmodel_1, scope = list(lower = nullmodel_1, upper = fullmodel_1), direction = "both")
summary(model_3_1)
#Linear Regression model with best parameters
best_reg <- lm(Today_positive_cases ~ . -Today_hospitalized -Total_negative -Hospitalized_cummalative 
               -Total_in_ICU -Total_people_recovered , data = train_1)
summary(best_reg)

#predicting on test set using the best linear regression model
best_reg_test<- predict(best_reg, newdata = test_1)
regr.eval(test_1$Today_positive_cases,best_reg_test)

#Plot of performance of Linear Regression model
x<- 1:length(test_1$Today_positive_cases)
plot(x,test_1$Today_positive_cases, pch=20, col= 'black', main='Actual to predicted plot for Linear Regression',
     xlab='Index of the test value',ylab= 'Actual_Test_value' )
points(x, best_reg_test, lwd='2', col='red')

#Linear SVM model
set.seed(2020)
model_svm<- svm(Today_positive_cases~ .,data=train_1, kernel= 'linear')
summary(model_svm)
#tuning parameters 
tune_svm_lin<- tune(svm, Today_positive_cases~ .,data=train_1, kernel= "linear", ranges=list(cost=c(0.01,.1,1,5,10,100,500)))
best_para<- tune_svm_lin$best.parameters
summary(tune_svm_lin)
#plot for best cost 
plot(tune_svm_lin, main='Best cost for linear SVM')
#Model with tunes parameters
model_svm_1<- svm(Today_positive_cases~ .,data=train_1, kernel= 'linear',cost= best_para$cost)
summary(model_svm_1)
#prediction on test set
test_svm_lin<- predict(model_svm_1,test_1) 
head(test_svm_lin)
#performance check
regr.eval(test_1$Today_positive_cases,test_svm_lin)
#Plot of performance
x<- 1:length(test_1$Today_positive_cases)
plot(x,test_1$Today_positive_cases, pch=20, col= 'black', main='Actual to predicted plot for Linear Kernel SVM',
     xlab='Index of the test value',ylab= 'Actual_Test_value' )
points(x,test_svm_lin, lwd='2', col='blue')

#Radial Kernel SVM
set.seed(2020)
Radial_KernelSVM_tune<- tune(svm,Today_positive_cases~ ., data= train_1, kernel= 'radial', ranges= list(cost=c(0.01,.1,1,5), gamma=c(.01,.02)))
summary(Radial_KernelSVM_tune)
#plot for parameters
plot(Radial_KernelSVM_tune, main='Best Parameters for Radial SVM Kernel')
#Radial SVM kernel model with best parameters
Kernel_svm_model<- svm(Today_positive_cases~., data=train_1, kernel= 'radial',
                       cost= Radial_KernelSVM_tune$best.parameters$cost,
                       gamma= Radial_KernelSVM_tune$best.parameters$gamma)
summary(Kernel_svm_model)
#prediction on test set
pred_kernel_svm<- predict(Kernel_svm_model, test_1)
head(pred_kernel_svm)
#performance evaluation
regr.eval(test_1$Today_positive_cases,pred_kernel_svm)
#plot for kernel svm performance
x<- 1:length(test_1$Today_positive_cases)
plot(x,test_1$Today_positive_cases, pch=20, col= 'black', main='Actual to predicted plot for Radial Kernel SVM',
     xlab='Index of the test value',ylab= 'Actual_Test_value')
points(x, pred_kernel_svm, lwd='2', col='green')


#KNN algorith 
#model with best parameters
set.seed(2020)
ctrl <- trainControl(method="repeatedcv",repeats = 3) #,classProbs=TRUE,summaryFunction = twoClassSummary)
knnFit <- train(Today_positive_cases ~ ., data = train_1, method = "knn", trControl = ctrl, preProcess = c("center","scale"), tuneLength = 8)
plot(knnFit)
#predicting on test set
knnpredict<- predict(knnFit, newdata= test_1)
head(knnpredict)
#performance evaluation
regr.eval(test_1$Today_positive_cases,knnpredict)
#Performance of KNN
x<- 1:length(test_1$Today_positive_cases)
plot(x,test_1$Today_positive_cases, pch=20, col= 'black',main='Actual to predicted plot for K-nearest Neighbour',
     xlab='Index of the test value',ylab= 'Actual_Test_value')
points(x, knnpredict, lwd='2', col='purple')
