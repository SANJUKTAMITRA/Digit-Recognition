############################ SVM Letter Recogniser #################################
# 1. Business Understanding
# 2. Data Understanding
# 3. Data Preparation
# 4. Model Building 
#  4.1 Linear kernel
#  4.2 RBF Kernel
# 5 Hyperparameter tuning and cross validation

#####################################################################################
#####################################################################################

# 1. Business Understanding: 
#The objective is to identify an image of a digit from 0 to 9 submitted by a user via a scanner,
#a tablet, or other digital device.Here we shall use  Support Vector Machine learning model for this purpose
#####################################################################################
#2. Data Understanding 

#MNIST data consists on training data and testing data
#training data
# Number of Instances: 60,000
# Number of Attributes: 785
#test or unseen data
# Number of Instances: 10,000
# Number of Attributes: 785

#####################################################################################
##3. Data Preparation: 


#Loading Neccessary libraries
library(ggplot2)
library(caTools)
library(kernlab)
library(caret)
library(e1071)
library(mlr)

install.packages("ggplot2")
install.packages("kernlab")
install.packages("caret")
install.packages("e1071")
install.packages("mlr")
#import the training data 

digit_recg <- read.csv("mnist_train.csv",stringsAsFactors = FALSE,header = TRUE)

#Import the mnist test data
digit_test <- read.csv("mnist_test.csv",stringsAsFactors = FALSE,header = TRUE)

#data preparation steps should be same for both test data and training data 
#Understanding Dimensions

dim(digit_recg)
dim(digit_test)

#Structure of the dataset

str(digit_recg)
str(digit_test)

#printing first few rows

head(digit_recg)
head(digit_test)

#Checking for missing values
sum(is.na(digit_recg))
sapply(digit_recg, function(x) sum(is.na(x)))
#no missing values for training data

sum(is.na(digit_test))
#no missing values for test data as well

#renaming the first column for both test data and training data
colnames(digit_recg)[1] <- "label"
#colnames(digit_test)[1] <- "handwritten_letter"
#Making our target class to factor

digit_recg$label <-as.factor(digit_recg$label)
#digit_test$handwritten_letter<-as.factor(digit_test$handwritten_letter)
str(digit_recg)
#str(digit_test)

#change all the pixel data to numeric 

digit_recg[,c(2:785)] <- sapply(digit_recg[,c(2:785)],function(x) as.numeric(x))

#checking for outliers 
#since this is a pixel data hence checking of usual outliers using quantile function may not work as it might result in 
#loss of an image or pixel data;cannot check for bad pixel data using quantile function.

#lets check summary of traindata

d <- summary(digit_recg)
View(d)
#After checking the summary of the dataframe digit_recg it is evident all the columns have ranges between minum value 0 and 
#maximum value 255. No column is exceeding 255.Hence no outliers present,


 #EDA 
digit_eda <- digit_recg
#plot image of any random number,first creating a matrix of 28 by 28 and then using image function to plot the number
#8
m1 <- matrix(as.numeric(digit_eda[56,-1]),nrow=28)
image(m1,col=heat.colors(55))
#6 inverse
m2 <- matrix(as.numeric(digit_eda[67,-1]),nrow=28)
image(m2,col=heat.colors(255))


#calculate average intensity of each number 0 to 9 in a row , first calcuate average mean of pixels of each row
#and then aggregate for each number 
j <- c()
for(i in 1:nrow(digit_eda)){
  j[i] <- mean(as.numeric(digit_eda[i,-1]))
}

digit_eda$intensity <- j

#computitng the average intesnity of each digit 0 to 9
intensity <- aggregate(digit_eda$intensity,by=list(digit_eda$handwritten_letter),FUN=mean)



ggplot(intensity,aes(x=factor(Group.1),y=x)) + geom_bar(stat = "identity",fill="green")
 #plot shows number 0,8 have highest intensity followed by 2 and 3.
#4 and 7 has least intensity


ggplot(digit_eda,aes(x=intensity)) + geom_histogram(binwidth = 1.1) + facet_wrap(~handwritten_letter)
#intensity graph shows almost all the numbers intensity distribution in pixel are normally distributed,however for
#digit 4,5,7,8 and 9 there are slight variations in the graph from normal distribution.

#creating a subset of 20 rows from number 0 to 9
d1 <- sample.split(1:nrow(digit_eda),20)
data <- digit_recg[d1,]
#below function is written to display the image of 12 digits from 12 rows together
par(mfrow=c(4,3))
for(i in 1:12){
  m3 <- matrix(as.numeric(data[i,-1]),nrow=28)
  j <- image(m3,col=heat.colors(7))
}
#again same  function is executed to display the image of reamining 8 digits from the dataframe data together
par(mfrow=c(3,3))
for(i in 13:20){
  m3 <- matrix(rev(as.numeric(data[i,-1])),nrow=28)
  j <- image(m3,col=heat.colors(7))
}


#this completes EDA and Data preparation



#####################################################################################
#Model Building
# Splitting the data between train and test
#taking a 24% sample of the digit_recg data , This data will be used for model creation purpose.
set.seed(123)
indices <- sample.split(1:nrow(digit_recg),0.24*nrow(digit_recg))

DATA <- digit_recg[indices,]

#creatied a dataframe DATA that will consists of .24% of the digit_recg data and after that splitting this data into 
#70:30 ratio for model building and model evaluation


indices1 <- sample.split(1:nrow(DATA),0.7*nrow(DATA))
traindata <- DATA[indices1,]
testdata <- DATA[!indices1,]
#creating model on this traindata dataframe



#construction linear model first on training data with C=1
Model_linear <- ksvm(label~ ., data = traindata, scale = FALSE, kernel = "vanilladot")
print(Model_linear)

Eval_linear <- predict(Model_linear,testdata[,-1])

confusionMatrix(Eval_linear,testdata$label)

# sensitivity value for each digit is around 82% to 98% and  overall accuracy is 91% for COST as 1
#lets create the model again using RBF kernel and compare the accuracy

Model_rbf <- ksvm(label~ ., data = traindata, scale = FALSE, kernel = "rbfdot")
print(Model_rbf)
Eval_rbf <- predict(Model_rbf,testdata[,-1])
confusionMatrix(Eval_rbf,testdata$label)
#With RBF kernel accuracy is around 92 to 99%, overall accuracy is around 95%.

#lets execute linear kernel on train data again with Cost function as 0.1
Model_linear2 <- ksvm(label~ ., data = traindata, scale = FALSE, kernel = "vanilladot",C=0.1)
print(Model_linear2)

Eval_linear2 <- predict(Model_linear2,testdata[,-1])

confusionMatrix(Eval_linear2,testdata$label)

#with Cost function 0.1 accuracy acheived with linear kernel as 90% which is same as with C=1,sensitivity acheived for each digit was 84 to 97%
#hence RBF kernel has given the best performance with C=1
#####################################################################
#Hyperparameter tuning and Cross Validation - Radial - SVM 
######################################################################

# We will use the train and train control function from caret package to perform crossvalidation
set.seed(7)
#using the mlr package to fine tune the ksvm radial model 
install.packages("mlr")
library(mlr)
## Define the task for traindata
train.task = makeClassifTask(data = traindata, target = "label")
print(train.task)

#setting the Cost Parameter and Sigma Values
ps = makeParamSet(
makeDiscreteParam("C", values = c(1.0, 2.0,3.0,4.0,5.0)),
makeDiscreteParam("sigma", values = c(0.1,0.5,1,2,4))
)
ctrl = makeTuneControlGrid()

#iters value 5 implies the tuneparmas function will do 5 fold cross validation
rdesc = makeResampleDesc("CV", iters = 5L)

#running thr tuneparams with train.task for 5 fold cross validation with different combination of C and sigma values
res = tuneParams("classif.ksvm", task = train.task, resampling = rdesc, par.set = ps, control = ctrl,measure=acc)

res$x
res$y
print(res_grid)
res$opt.path
res_grid = as.data.frame(res$opt.path)
#tune result specify C=5 as the best solution , but we will select C=4 which has almost same and similar accracy as C=5

fit.svm <- tune.svm(label~., data=traindata, gamma = c(0.015 ,0.025, 0.05), cost = c(0.1,0.5,1,2))



######################################################################
# Validating the model results on  the test data 
######################################################################
#creating the final model_radial with C=4
model_radial <- ksvm(label~ ., data = traindata, scale = FALSE, kernel = "rbfdot",C=4)

#prediction and validation with confusion matrix
evaluate_final1<- predict(model_radial, testdata)
confusionMatrix(evaluate_final1, testdata$label)

# Accuracy    - 96.4%
#                      Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6 Class: 7 Class: 8 Class: 9
#Sensitivity           0.99662   0.9803  0.93091  0.93289  0.96610  0.97683  0.96721  0.95724  0.93226  0.91437
#Specificity           0.99487   0.9974  0.99454  0.99486  0.99084  0.99494  0.99706  0.99559  0.99484  0.99555



######################################################################
# Validating the model results on the mnist test data
######################################################################
digit_test$label <- digit_recg[28000,1]
digit_test <- digit_test[,c(785,1:784)]
digit_test$label <- as.factor(digit_test$label)

digit_test <- rbind(digit_recg[1,],digit_test)
digit_test <- digit_test[-1,]
evaluate_final2<- predict(model_radial, digit_test)
digit_test$label <- evaluate_final2

label_submission <- digit_test$label
write.csv(label_submission,"label.csv")
# Accuracy    - 96.5%
# Sensitivity - Class: 0  Class: 1  Class: 2  Class: 3  Class: 4  Class: 5  Class: 6  Class: 7  Class: 8   Class: 9
#               0.9898    0.9921    0.9545    0.9535    0.9695    0.9540    0.9760    0.9514    0.9528     0.9504
# Specificity - 0.9969     0.9973   0.9957    0.9951    0.9960    0.9960    0.9966    0.9964    0.9958     0.9950
######################################################################

