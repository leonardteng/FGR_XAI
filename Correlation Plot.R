# https://online.stat.psu.edu/stat462/node/180/
set.seed(7)
# load the library
library(mlbench)
library(caret)
library(VIF)
library(car)
library(dplyr)
library(ggcorrplot)
IUGR <- read.csv("IUGR_case_3.csv")
IUGR$labels <- factor(IUGR$labels)
IUGR$CM <- as.numeric(IUGR$CM)
IUGR$EFW <- as.numeric(IUGR$EFW)
IUGR$EFW <- as.numeric(IUGR$EFW)
IUGR <- IUGR[-c(1:2, 17)]

str(IUGR)
summary(IUGR)
# calculate correlation matrix - remove labels(predicted variable), age, ethnics
print(names(IUGR))
correlationMatrix <- cor(IUGR)
ggcorrplot(correlationMatrix)
# summarize the correlation matrix
table2 <- print(round(correlationMatrix,3))
library(xlsx)
write.table(table2, "mydata.txt", sep="\t")
