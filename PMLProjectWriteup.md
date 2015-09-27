---
title: "Practical Machine Learning Course Project"
author: "Abhijit Shetti"
date: "August 23, 2015"
output: html_document
---

## Introduction
There is a growing adoption of devices that measure physical movement using sensors attached to the human body. These devices allow one to track movement while one goes about their regular work or exercise. A large number of health conscious individuals are using these devices to meaure their physical activity when they exercise. 

While the devices track movement and report on it, they do not give any indication of how well the exercies is being done. In this project, we will use data collected from participants who used the devices while performing a specific exercise (dumbbell curl) both correctly and with five common mistakes individuals make while performing this exercise. The intent is to build a model that will predict from the device data whether the individual performed the exercise correctly or else the specific mistake they made.

## Load Data

We will use the data set from the study made available at http://groupware.les.inf.puc-rio.br/har. 

### Download Data
Lets get the data files and store them locally for later use.


```r
library(caret,quietly = TRUE)
library(rpart,quietly = TRUE)
library(rpart.plot,quietly = TRUE)
library(randomForest,quietly = TRUE)
library(corrplot,quietly = TRUE)
library(plyr,quietly = TRUE)
dataDir <- "./Data"
trainUrl <-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
trainFile <- paste0(dataDir,"/pml-training.csv")
testFile  <- paste0(dataDir,"/pml-testing.csv")
if (!file.exists(dataDir)) {
  dir.create(dataDir)
}
if (!file.exists(trainFile)) {
  download.file(trainUrl, destfile=trainFile, method="curl")
}
if (!file.exists(testFile)) {
  download.file(testUrl, destfile=testFile, method="curl")
}
```

### Read the data files
Lets read in the data files into a data frame and get a summary of what they contain


```r
trainData <- read.csv("./data/pml-training.csv",header=TRUE)
testData <- read.csv("./data/pml-testing.csv",header=TRUE)
```

We use read.csv to load the data into a data frame object. Note the stringsAsFactors argument - we do not want any string data to be read as Factors yet.

### Explore and clean the data
Lets compare the column names in each data set and see which ones are not equal

```r
print(paste0("Test Data Columns: ",length(colnames(testData))," Train Data Columns: ",
             length(colnames(trainData))))
```

```
## [1] "Test Data Columns: 160 Train Data Columns: 160"
```

```r
identical(colnames(testData),colnames(trainData))
```

```
## [1] FALSE
```

```r
diffColumns <- which(colnames(trainData)!=colnames(testData))
colnames(testData)[diffColumns]
```

```
## [1] "problem_id"
```

```r
colnames(trainData)[diffColumns]
```

```
## [1] "classe"
```

We see the last column is named differently in the two data sets. This is our outcome column, which we shall be predicting with our chosen machine learning algorithm.
Lets first preprocess the data and select our feature set to make it ready for analysis.
Lets remove columns with NA values

```r
nmissing <- function(x) sum(is.na(x))
colwiseMissing <- colwise(nmissing)(trainData)
sparseColIndices <- which(colwiseMissing > 0)
sparseCols <- colnames(trainData)[sparseColIndices]
noNAtrainData <- trainData[,!(colnames(trainData) %in% sparseCols)]
noNAtestData <- testData[,!(colnames(testData) %in% sparseCols)]
```

Next, lets remove near zero variables from the data set. These will not add to the model accuracy.

```r
nzv <- nearZeroVar(noNAtrainData,saveMetrics = TRUE)
nearZeroCols <- rownames(nzv)[nzv$nzv==TRUE]
nonZeroNATrainData <- noNAtrainData[,!(colnames(noNAtrainData) %in% nearZeroCols)]
nonZeroNATestData <- noNAtestData[,!(colnames(noNAtestData) %in% nearZeroCols)]
```

## Model selection and data slicing

We shall try a couple of different models on the data given. Since the outcomes are categorical, we shall select from among models best used for classification rather than regression. One of the most commonly used models is Random Forest. Lets try a couple of different models with different parameters and select from among them based on the model performance. We will use cross validation within each model fit to ensure we dont overfit.  

To do this, we will use the createFolds method to create two folds each to serve as the training and test sets.


```r
set.seed(1212)
trainFolds <- createFolds(y=nonZeroNATrainData$classe,k=2,returnTrain=TRUE)
testFolds <- createFolds(y=nonZeroNATrainData$classe,k=2,returnTrain=FALSE) 
```

## Model fit and evaluation

First we will apply Random Forest with the first fold and 50 trees.


```r
data <- nonZeroNATrainData[trainFolds$Fold1,]
modelFit1 <- train(data$classe~.,method="rf", ntree=50, trControl=trainControl(method="cv", 
                  number=10), data=data[,-c(1:6,59)])
```

Lets check the model fit.


```r
plot(modelFit1$finalModel, main= "Error Rates")
```

![plot of chunk unnamed-chunk-8](figure/unnamed-chunk-8-1.png) 


Now lets apply Random Forest with the second fold and 100 trees with 5 fold cross validation


```r
data <- nonZeroNATrainData[trainFolds$Fold2,]
modelFit2 <- train(data$classe~.,method="rf", ntree=100, trControl=trainControl(method="cv", number=5), data=data[,-c(1:6,59)])
```

Lets check the model fit.

```r
plot(modelFit2$finalModel, main= "Error Rates")
```

![plot of chunk unnamed-chunk-10](figure/unnamed-chunk-10-1.png) 


## Chosen Model Out of Sample Error Estimate
We will use the first model as it already has very low error rates, and takes a much smaller time to finish.
Lets print the confusion matrix to see the individual exercise class error and the overall accuracy and error rate.

```r
testData<-nonZeroNATrainData[testFolds$Fold1,]
conMatrix <- confusionMatrix(testData$classe,predict(modelFit1,testData))
print(conMatrix$table)
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 2788    0    2    0    0
##          B    7 1882    9    0    0
##          C    0    5 1705    1    0
##          D    0    0   12 1594    2
##          E    0    1    0    2 1800
```

```r
print(conMatrix$overall)
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.9958206      0.9947134      0.9943344      0.9969992      0.2849134 
## AccuracyPValue  McnemarPValue 
##      0.0000000            NaN
```

Our estimated accuracy is 99.6% and the out of sample error estimate is 0.4%.

## Results
Lets apply the model fit above to our test data.

```r
predict(modelFit1,nonZeroNATestData)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

This is the resultant prediction on the test data which we shall submit as our predicted values.
