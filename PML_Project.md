Practical Machine Learning - Course Project
========================================================

Load the required libraries to building the Machine Learning Algorithm.


```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(randomForest)
```

```
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r
library(knitr)
```


Set seed to 12345.


```r
set.seed(12345)
```


Load training data set.


```r
data <- read.csv("pml-training.csv", header = TRUE)
```


Take the necessary processing steps to create a suitable training data set.


```r
## Convert classe into factor
data$classe <- as.factor(data$classe)

## Create partitions in training data
inTrain <- createDataPartition(y = data$classe, p = 0.75, list = FALSE)

## Split the training set into training set into training and cross
## validation sets
training <- data[inTrain, ]
crossVal <- data[-inTrain, ]

## Determine the predictors that have more than 25% empty or NA
dropNA <- c()
dropEmpty <- c()

for (i in 1:dim(training)[2]) {
    if (length(training[, i][is.na(training[, i])]) > 0.25 * dim(training)[2]) {
        dropNA <- c(dropNA, colnames(training)[i])
    }
}

for (i in 1:dim(training)[2]) {
    if (length(which(training[, i] == "")) > 0.25 * dim(training)[2]) {
        dropEmpty <- c(dropEmpty, colnames(training)[i])
    }
}

drop <- c(dropNA, dropEmpty[!(dropEmpty %in% dropNA)])

## Drop the first seven columns from the training and cross validation sets
training <- training[, 8:length(names(training))]
crossVal <- crossVal[, 8:length(names(crossVal))]

## Drop the predictors that have more than 25% NA or empty cells
training <- training[, !(names(training) %in% drop)]
crossVal <- crossVal[, !(names(crossVal) %in% drop)]
```


Train a prediction model using the Random Forest algorithm.


```r
## Train using Random Forest algorithm and all remaining predictors
modFit <- train(classe ~ ., data = training, method = "rf", tuneGrid = data.frame(mtry = 3), 
    trControl = trainControl(method = "none"))
modFit
```

```
## Random Forest 
## 
## 14718 samples
##    52 predictors
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: None
```


Cross validate the model by predicting using the cross validation set and tabulate the results to determine out of sample prediction accuracy.


```r
## Predict for 'classe' using cross validation set and compare results with
## actual outcome
pred <- predict(modFit, crossVal)
crossVal$predRight <- pred == crossVal$classe
table(pred, crossVal$classe)
```

```
##     
## pred    A    B    C    D    E
##    A 1394    8    0    0    0
##    B    1  937    3    0    0
##    C    0    4  852    9    2
##    D    0    0    0  795    4
##    E    0    0    0    0  895
```


Expected out of sample accuracy is 99.4%.


```r
crossVal$predWrong <- pred != crossVal$classe
sum(crossVal$predRight)/sum(crossVal$predRight, crossVal$predWrong)
```

```
## [1] 0.9937
```


Load testing data and process to reflect changes to training data.


```r
testing <- read.csv("pml-testing.csv", header = TRUE)

## Remove columns removed from training set
testing <- testing[, 8:length(names(testing))]
testing <- testing[, !(names(testing) %in% drop)]
```


Predict outcomes for test set.


```r
pred_test <- predict(modFit, testing)
```

