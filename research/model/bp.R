#!/usr/bin/env Rscript


library(neuralnet)
source('lib/utils.R')
set.seed(2)

samples <- read.table('../data/57_0811-0821-sold-tbsold-rate.ssv', header=TRUE)
normed <- miner.scale(samples)
splited <- miner.sample(normed, 0.85)

model <- neuralnet(sold ~ click + tbsold + ratenum, splited$main, hidden=c(3,2), linear.output=TRUE)

plot(model)
pred <- compute(model, splited$remain[, c('click', 'tbsold', 'ratenum')])
sold.min <- min(samples$sold)
sold.norm <- max(samples$sold) - sold.min
pred.unnormed <- pred$net.result * sold.norm + sold.min
samples.test.sold <- splited$remain$sold * sold.norm + sold.min
rownames(pred.unnormed) <- 1:nrow(pred.unnormed)
MSE <- sum((samples.test.sold - pred.unnormed[,1]) ^ 2) / length(samples.test.sold)

par(mfrow=c(1,2))
plot(samples.test.sold, pred.unnormed, col='red', main='samples vs predict', pch=18, cex=0.7)
abline(0, 1, lwd=2)


library(RSNNS)
inputs <- as.matrix(seq(0,10,0.1))
outputs <- as.matrix(sin(inputs) + runif(inputs*0.2))
outputs <- normalizeData(outputs, "0_1")
model <- rbf(inputs, outputs, size=40, maxit=1000,
             initFuncParams=c(0, 1, 0, 0.01, 0.01),
             learnFuncParams=c(1e-8, 0, 1e-8, 0.1, 0.8),
             linOut=TRUE)
par(mfrow=c(2,1))
plotIterativeError(model)
plot(inputs, outputs)
lines(inputs, fitted(model), col="green")

RBF_Weights_Kohonen(0,0,0,0,0) 

