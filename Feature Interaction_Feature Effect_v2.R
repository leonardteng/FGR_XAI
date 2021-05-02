#https://rdrr.io/cran/iml/man/Interaction.html
#https://books.google.com.my/books?id=QdW8DwAAQBAJ&pg=SA14-PA37&lpg=SA14-PA37&dq=2-way+interaction+strength+using+iml&source=bl&ots=6SUCNZEshf&sig=ACfU3U1D1faD9NbVVrJ_ze1vcaVCIZFZew&hl=en&sa=X&ved=2ahUKEwjH5KTltd7oAhWkyDgGHRiaAQwQ6AEwBHoECAwQKQ#v=onepage&q=2-way%20interaction%20strength%20using%20iml&f=false
#https://christophm.github.io/interpretable-ml-book/interaction.html

#Most relevant
#https://cran.r-project.org/web/packages/iml/vignettes/intro.html
#https://www.exago.ml/blog/2019/4/28/model-explainability-with-keras-and-iml
#https://cran.r-project.org/web/packages/iml/vignettes/intro.html
#https://www.shirin-glander.de/2018/07/explaining_ml_models_code_caret_iml/
#http://uc-r.github.io/iml-pkg


IUGR <- read.csv("IUGR_case_3.csv")

### find pairwise interactions
#library("rpart")
library(iml)
library("randomForest")
library(e1071)
library("caret")
library(caTools)
library(tibble)

library(reticulate)

library("ggplot2")

library(dplyr)


set.seed(131)
# Fit a CART on IUGR dataset
#rf <- rpart(labels ~ ., data = IUGR)

print((table(IUGR$labels)))
print(prop.table(table(IUGR$labels)))
IUGR$labels <- factor(IUGR$labels)
#levels(IUGR$labels) <- list(no="0", yes="1")
print(table(IUGR$labels))

split <- sample.split(IUGR$labels, SplitRatio = 0.75)
trainSplit <- subset(IUGR, split == TRUE)
testSplit <- subset(IUGR, split == FALSE)
as.data.frame(table(trainSplit$labels))

library("ROSE")
trainSplit_balanced <- ovun.sample(labels ~ ., data = trainSplit, method = "over",N = 218)$data
as.data.frame(table(trainSplit_balanced$labels))

#train_control <- trainControl(method = "cv", number=10, classProbs = TRUE)
train_control <- trainControl(method = "cv", number=10)
#svm_model <- caret::train(labels ~., data= trainSplit_balanced, method = "svmRadial", trcontrol = train_control)
svm_model <- caret::train(labels ~., data= trainSplit_balanced, method = "svmRadial", trcontrol = train_control, preProcess = c("scale","center"))
svm_model

#Test how good the model is
test_predict <- predict(svm_model, testSplit)
#predict(svm_model, newdata = testSplit)
confusionMatrix(test_predict, trainSplit$labels)
test_performance <- testSplit %>% tibble::as_tibble() %>% dplyr::select(labels) %>% tibble::add_column(prediction=as.vector(test_predict)) %>% mutate(correct = ifelse(labels ==prediction, "correct", "wrong")) %>% mutate_if(is.character, as.factor)

# Create a model object
#mod <- Predictor$new(svm_model, data = testSplit[-which(names(testSplit) == "labels")], y = testSplit$labels)
mod <- Predictor$new(svm_model, data = testSplit[-which(names(testSplit) == "labels")], y = testSplit$labels)

# Measure the interaction strength
ia <- Interaction$new(mod)

# All-way H-statistic
ia$results %>% arrange(desc(.interaction)) %>%
  top_n(10)

# Plot the resulting leaf nodes
plot(ia)

# feature of interest
#feat <- "Nuchal.fold"
#feat <- "GA..wk."
#feat <- "Femur"
#feat <- "Cerebellum.tr"
#feat <- "BPD"
#feat <- "AC"
#feat <- 'Va'
#feat <- "Vp"
#feat <- "HC"
#feat <- "Hem"
#feat <- "Ut.RI"
feat <- "Ut.PI"

# 2-way H-statistic
ia_2way <- Interaction$new(mod, feat)
ia_2way$results %>% arrange(desc(.interaction)) %>% top_n(10)

# ALE PLOT
ale <- FeatureEffect$new(mod, feature ="Ut.PI", method = "ale")
ale$plot()

#LIME
features <- testSplit[,1:16]
glimpse(features)
nuchal_fold_cutoff_1 <- features[features$Nuchal.fold == 3.7,]  
nuchal_fold_cutoff_2 <- features[features$Nuchal.fold == 3.73,]  
lime.svm_3.7 <- LocalModel$new(mod, k=10, x.interest = nuchal_fold_cutoff_1) %>% plot()+ ggtitle("Nuchal Fold of 3.7")
lime.svm_3.73 <- LocalModel$new(mod, k=10, x.interest = nuchal_fold_cutoff_2) %>% plot()+ ggtitle("Nuchal Fold of 3.73")
gridExtra::grid.arrange(lime.svm_3.7, lime.svm_3.73, nrow = 1)


# To split into wrong classified and correct classified row
testSplit_df <-  as.data.frame(testSplit)

test_Split_2 <-  testSplit_df %>%
  as.data.frame() %>% 
  mutate(sample_id = rownames(testSplit))

test_correct <- test_performance %>% 
  mutate(sample_id = rownames(test_Split_2)) %>% 
  filter(correct == 'correct') %>%
  inner_join(test_Split_2)

test_wrong <- test_performance %>% 
  mutate(sample_id = rownames(test_Split_2)) %>% 
  filter(correct == 'wrong') %>%
  inner_join(test_Split_2)

write.csv(test_wrong,"test_Wrong.csv", row.names = FALSE)
write.csv(test_correct,"test_correct.csv", row.names = FALSE)

test_correct_1 <- test_performance %>% 
  mutate(sample_id = rownames(test_Split_2)) %>% 
  filter(correct == 'correct') %>%
  inner_join(test_Split_2) %>% 
  dplyr::select(-c(prediction, correct, sample_id))

test_wrong_1 <- test_performance %>% 
  mutate(sample_id = rownames(test_Split_2)) %>% 
  filter(correct == 'wrong') %>%
  inner_join(test_Split_2) %>% 
  dplyr::select(-c(prediction, correct, sample_id))

Wrong_classified_1 <- test_wrong_1[1,-1]
lime.svm_wrong_1 <- LocalModel$new(mod, k=16, x.interest = Wrong_classified_1) %>% plot()
Wrong_classified_2 <- test_wrong_1[2,-1]
lime.svm_wrong_2 <- LocalModel$new(mod, k=16, x.interest = Wrong_classified_2) %>% plot()
Wrong_classified_3 <- test_wrong_1[3,-1]
lime.svm_wrong_3 <- LocalModel$new(mod, k=16, x.interest = Wrong_classified_3) %>% plot()
gridExtra::grid.arrange(lime.svm_wrong_1, lime.svm_wrong_2, lime.svm_wrong_3, nrow = 3)

Wrong_classified_4 <- test_wrong_1[4,-1]
lime.svm_wrong_4 <- LocalModel$new(mod, k=16, x.interest = Wrong_classified_4) %>% plot()
Wrong_classified_5 <- test_wrong_1[5,-1]
lime.svm_wrong_5 <- LocalModel$new(mod, k=16, x.interest = Wrong_classified_5) %>% plot()
Wrong_classified_6 <- test_wrong_1[6,-1]
lime.svm_wrong_6 <- LocalModel$new(mod, k=16, x.interest = Wrong_classified_6) %>% plot()
gridExtra::grid.arrange(lime.svm_wrong_4, lime.svm_wrong_5, lime.svm_wrong_6, nrow = 3)

Wrong_classified_7 <- test_wrong_1[7,-1]
lime.svm_wrong_7 <- LocalModel$new(mod, k=16, x.interest = Wrong_classified_7) %>% plot()
Wrong_classified_8 <- test_wrong_1[8,-1]
lime.svm_wrong_8 <- LocalModel$new(mod, k=16, x.interest = Wrong_classified_8) %>% plot()
Wrong_classified_9 <- test_wrong_1[9,-1]
lime.svm_wrong_9 <- LocalModel$new(mod, k=16, x.interest = Wrong_classified_9) %>% plot()
gridExtra::grid.arrange(lime.svm_wrong_7, lime.svm_wrong_8, lime.svm_wrong_9, nrow = 3)

Wrong_classified_10 <- test_wrong_1[10,-1]
lime.svm_wrong_10 <- LocalModel$new(mod, k=16, x.interest = Wrong_classified_10) %>% plot()
Wrong_classified_11 <- test_wrong_1[11,-1]
lime.svm_wrong_11 <- LocalModel$new(mod, k=16, x.interest = Wrong_classified_11) %>% plot()
Wrong_classified_12 <- test_wrong_1[12,-1]
lime.svm_wrong_12 <- LocalModel$new(mod, k=16, x.interest = Wrong_classified_12) %>% plot()
gridExtra::grid.arrange(lime.svm_wrong_10, lime.svm_wrong_11, lime.svm_wrong_12, nrow = 3)

Wrong_classified_13 <- test_wrong_1[13,-1]
lime.svm_wrong_13 <- LocalModel$new(mod, k=16, x.interest = Wrong_classified_13) %>% plot()
Wrong_classified_14 <- test_wrong_1[14,-1]
lime.svm_wrong_14 <- LocalModel$new(mod, k=16, x.interest = Wrong_classified_14) %>% plot()
gridExtra::grid.arrange(lime.svm_wrong_13, lime.svm_wrong_14, nrow = 2)


#test correct using LIME

correct_classified_1 <- test_correct_1[1,-1]
lime.svm_correct_1 <- LocalModel$new(mod, k=16, x.interest = correct_classified_1) %>% plot()
correct_classified_2 <- test_correct_1[2,-1]
lime.svm_correct_2 <- LocalModel$new(mod, k=16, x.interest = correct_classified_2) %>% plot()
correct_classified_3 <- test_correct_1[3,-1]
lime.svm_correct_3 <- LocalModel$new(mod, k=16, x.interest = correct_classified_3) %>% plot()
gridExtra::grid.arrange(lime.svm_correct_1, lime.svm_correct_2, lime.svm_correct_3, nrow = 3)

correct_classified_4 <- test_correct_1[4,-1]
lime.svm_correct_4 <- LocalModel$new(mod, k=16, x.interest = correct_classified_4) %>% plot()
correct_classified_5 <- test_correct_1[5,-1]
lime.svm_correct_5 <- LocalModel$new(mod, k=16, x.interest = correct_classified_5) %>% plot()
correct_classified_6 <- test_correct_1[6,-1]
lime.svm_correct_6 <- LocalModel$new(mod, k=16, x.interest = correct_classified_6) %>% plot()
gridExtra::grid.arrange(lime.svm_correct_4, lime.svm_correct_5, lime.svm_correct_6, nrow = 3)



# to run with label and prediction =1
correct_classified_10 <- test_correct_1[10,-1]
lime.svm_correct_10 <- LocalModel$new(mod, k=16, x.interest = correct_classified_10) %>% plot()
correct_classified_11 <- test_correct_1[11,-1]
lime.svm_correct_11 <- LocalModel$new(mod, k=16, x.interest = correct_classified_11) %>% plot()
correct_classified_26 <- test_correct_1[26,-1]
lime.svm_correct_26 <- LocalModel$new(mod, k=16, x.interest = correct_classified_26) %>% plot()
gridExtra::grid.arrange(lime.svm_correct_10, lime.svm_correct_11, lime.svm_correct_26, nrow = 3)

#test wrong using shapley

shapley_svm_wrong_1 <- Shapley$new(mod, x.interest = test_wrong_1[1,-1]) %>% plot()
shapley_svm_wrong_2 <- Shapley$new(mod, x.interest = test_wrong_1[2,-1]) %>% plot()
shapley_svm_wrong_3 <- Shapley$new(mod, x.interest = test_wrong_1[3,-1]) %>% plot()
gridExtra::grid.arrange(shapley_svm_wrong_1, shapley_svm_wrong_2, shapley_svm_wrong_3, nrow = 3)

shapley_svm_wrong_4 <- Shapley$new(mod, x.interest = test_wrong_1[4,-1]) %>% plot()
shapley_svm_wrong_5 <- Shapley$new(mod, x.interest = test_wrong_1[5,-1]) %>% plot()
shapley_svm_wrong_6 <- Shapley$new(mod, x.interest = test_wrong_1[6,-1]) %>% plot()
gridExtra::grid.arrange(shapley_svm_wrong_4, shapley_svm_wrong_5, shapley_svm_wrong_6, nrow = 3)

shapley_svm_wrong_7 <- Shapley$new(mod, x.interest = test_wrong_1[7,-1]) %>% plot()
shapley_svm_wrong_8 <- Shapley$new(mod, x.interest = test_wrong_1[8,-1]) %>% plot()
shapley_svm_wrong_9 <- Shapley$new(mod, x.interest = test_wrong_1[9,-1]) %>% plot()
gridExtra::grid.arrange(shapley_svm_wrong_7, shapley_svm_wrong_8, shapley_svm_wrong_9, nrow = 3)

shapley_svm_wrong_10 <- Shapley$new(mod, x.interest = test_wrong_1[10,-1]) %>% plot()
shapley_svm_wrong_11 <- Shapley$new(mod, x.interest = test_wrong_1[11,-1]) %>% plot()
shapley_svm_wrong_12 <- Shapley$new(mod, x.interest = test_wrong_1[12,-1]) %>% plot()
gridExtra::grid.arrange(shapley_svm_wrong_10, shapley_svm_wrong_11, shapley_svm_wrong_12, nrow = 3)

shapley_svm_wrong_13 <- Shapley$new(mod, x.interest = test_wrong_1[13,-1]) %>% plot()
shapley_svm_wrong_14 <- Shapley$new(mod, x.interest = test_wrong_1[14,-1]) %>% plot()
gridExtra::grid.arrange(shapley_svm_wrong_13, shapley_svm_wrong_14, nrow = 2)

#test correct using shapley
shapley_svm_correct_1 <- Shapley$new(mod, x.interest = test_correct_1[1,-1]) %>% plot()
shapley_svm_correct_2 <- Shapley$new(mod, x.interest = test_correct_1[2,-1]) %>% plot()
shapley_svm_correct_3 <- Shapley$new(mod, x.interest = test_correct_1[3,-1]) %>% plot()
gridExtra::grid.arrange(shapley_svm_correct_1, shapley_svm_correct_2, shapley_svm_correct_3, nrow = 3)

shapley_svm_correct_4 <- Shapley$new(mod, x.interest = test_correct_1[4,-1]) %>% plot()
shapley_svm_correct_5 <- Shapley$new(mod, x.interest = test_correct_1[5,-1]) %>% plot()
shapley_svm_correct_6 <- Shapley$new(mod, x.interest = test_correct_1[6,-1]) %>% plot()
gridExtra::grid.arrange(shapley_svm_correct_4, shapley_svm_correct_5, shapley_svm_correct_6, nrow = 3)

# to run with label and prediction =1
shapley_svm_correct_10 <- Shapley$new(mod, x.interest = test_correct_1[10,-1]) %>% plot()
shapley_svm_correct_11 <- Shapley$new(mod, x.interest = test_correct_1[11,-1]) %>% plot()
shapley_svm_correct_26 <- Shapley$new(mod, x.interest = test_correct_1[26,-1]) %>% plot()
gridExtra::grid.arrange(shapley_svm_correct_10, shapley_svm_correct_11, shapley_svm_correct_26, nrow = 3)

shapley_svm_correct_33 <- Shapley$new(mod, x.interest = test_correct_1[33,-1]) %>% plot()
shapley_svm_correct_34 <- Shapley$new(mod, x.interest = test_correct_1[34,-1]) %>% plot()
shapley_svm_correct_35 <- Shapley$new(mod, x.interest = test_correct_1[35,-1]) %>% plot()
gridExtra::grid.arrange(shapley_svm_correct_33, shapley_svm_correct_34, shapley_svm_correct_35, nrow = 3)

gridExtra::grid.arrange(shapley_svm_correct_1, nrow = 1)
gridExtra::grid.arrange(shapley_svm_correct_2, nrow = 1)
gridExtra::grid.arrange(shapley_svm_correct_3, nrow = 1)
gridExtra::grid.arrange(shapley_svm_correct_4, nrow = 1)
gridExtra::grid.arrange(shapley_svm_correct_5, nrow = 1)
gridExtra::grid.arrange(shapley_svm_correct_6, nrow = 1)
gridExtra::grid.arrange(shapley_svm_correct_10, nrow = 1)
gridExtra::grid.arrange(shapley_svm_correct_11, nrow = 1)
gridExtra::grid.arrange(shapley_svm_correct_26, nrow = 1)
gridExtra::grid.arrange(shapley_svm_correct_33, nrow = 1)
gridExtra::grid.arrange(shapley_svm_correct_35, nrow = 1)


