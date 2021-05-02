#https://mixomicsteam.github.io/Bookdown/plsda.html
#http://mixomics.org/methods/pls-da/

#if (!requireNamespace("BiocManager", quietly = TRUE))
install.packages("BiocManager")

BiocManager::install("mixOmics")

## install BiocManager if not installed
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
## install mixOmics
BiocManager::install('mixOmics')
library('mixOmics')
IUGR <- read.csv("IUGR_case_3.csv")
IUGR$labels <- factor(IUGR$labels)
IUGR$CM <- as.numeric(IUGR$CM)
IUGR$EFW <- as.numeric(IUGR$EFW)

IUGR_2 <- subset(IUGR)
str(IUGR_2)
X <- subset(IUGR_2, select = -c(labels))
y <- IUGR_2$labels
plsda.res = plsda(X, y, ncomp=5)

# this code takes ~ 1 min to run
set.seed(2543) # for reproducibility here, only when the `cpus' argument is not used
perf.plsda <- perf(plsda.res, validation = "Mfold", folds = 10, progressBar = FALSE, auc = TRUE, nrepeat = 10) 
perf.plsda$error.rate  # error rates
# break down of error rate per class is also insightful on the prediction
# of the model:
perf.plsda$error.rate.class
plot(perf.plsda, col = color.mixo(1:3), sd = TRUE, legend.position = "horizontal") #ncomp =2 seems to achieve best classification model

#PLSDA.VIP(perf.plsda)
## sPLS-DA for variable selection
# Tuning sPLS-DA

list.keepX <- c(1:14)
list.keepX

set.seed(30)
tune.splsda.srbct <- tune.splsda(X, y, ncomp = 2, # use 2 since figure plot show 2
                                 validation = 'Mfold',
                                 folds = 10, dist = 'max.dist', progressBar = FALSE,
                                 measure = "BER", test.keepX = list.keepX,
                                 nrepeat = 10)   # we suggest nrepeat = 50

error <- tune.splsda.srbct$error.rate
error
tune.splsda.srbct$choice.keepX           #The optimal number of features to select (per component)      

choice.ncomp <- tune.splsda.srbct$choice.ncomp$ncomp # optimal number of components based on t-tests on the error rate
choice.ncomp

choice.keepX <- tune.splsda.srbct$choice.keepX[1:choice.ncomp]  # optimal number of variables to select
choice.keepX


# Include best parameters in final sPLSDA model.
## sPLS-DA function
#splsda.res <- splsda(X, y, ncomp = choice.ncomp, keepX = choice.keepX) # where keepX is the number of variables selected for each components
splsda.res <- splsda(X, y, ncomp = choice.ncomp, keepX = 16) # where keepX is the number of variables selected for each components

# The performance of our final sPLSDA model
perf.splsda <- perf(splsda.res, validation = "Mfold", folds = 5, 
                    progressBar = FALSE, auc = TRUE, nrepeat = 10) 

perf.splsda$error.rate

# break down of error rate per class is also insightful on the prediction
# of the model:
perf.splsda$error.rate.class

# Final selection of features can be output, along with their weight coefficient(most important based on their absolute value)
# using selectVar function:
selectVar(splsda.res, comp = 1)$value
