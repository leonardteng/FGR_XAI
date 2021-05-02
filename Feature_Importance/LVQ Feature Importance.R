set.seed(7)
# load the library
library(mlbench)
library(caret)
library(lattice)
library(ggplot2)
# load the dataset
IUGR <- read.csv("IUGR_case_3.csv")

IUGR$labels <- factor(IUGR$labels)
IUGR$CM <- as.numeric(IUGR$CM)
IUGR$EFW <- as.numeric(IUGR$EFW)

summary(IUGR)
# prepare training scheme
control <- trainControl(method="cv", number=10)
# train the model
model <- train(labels~., data=IUGR, method="lvq", preProcess="scale", trControl=control)
# estimate variable importance
importance <- varImp(model, scale=FALSE)
# summarize importance
print(importance)
# plot importance
plot(importance)


