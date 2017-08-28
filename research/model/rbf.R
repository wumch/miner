#!/usr/bin/env Rscript

library(RSNNS)
source('lib/utils.R')

names_ <- names(train_data)
formula_ <- paste(c(names_[1], paste(names_[2:length(names_)], collapse = " + ")), collapse=' ~ ')
f <- as.formula(formula_)


set.seed(2)
samples <- read.table('../data/57_0811-0821-sold-tbsold-rate.ssv', header=TRUE)
normed <- miner.scale(samples)
normed.target <- normed$sold
normed.input <- normed[, 2:ncol(normed)]
splited <- splitForTrainingAndTest(normed.input, normed.target, ratio=0.15)

model <- rbf(splited$inputsTrain, splited$targetsTrain, 
             size=40, maxit=1000, linOut=TRUE)

summary(model)
par(mfrow=c(2,1))
plotIterativeError(model)

pred <- predict(model, splited$inputsTest)
plot(pred, splited$targetsTest)
