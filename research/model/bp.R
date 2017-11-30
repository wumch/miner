#!/usr/bin/env Rscript


library(neuralnet)
source('lib/utils.R')
set.seed(7)

samples <- read.table('../data/t121_ctr_tbsold.tsv', header=TRUE)
normed <- miner.scale(samples)
splited <- miner.sample(normed, 0.85)

model <- neuralnet(ctr ~ paid + price + coupon_price + profit_ratio + tbsold + ratenum, splited$main, hidden=c(4,2), linear.output=TRUE)

plot(model)
pred <- compute(model, splited$remain[, c('paid', 'price', 'coupon_price', 'profit_ratio', 'tbsold', 'ratenum')])
ctr.min <- min(samples$ctr)
ctr.norm <- max(samples$ctr) - ctr.min
pred.unnormed <- pred$net.result * ctr.norm + ctr.min
samples.test.ctr <- splited$remain$ctr * ctr.norm + ctr.mini
rownames(pred.unnormed) <- 1:nrow(pred.unnormed)
MSE <- sum((samples.test.ctr - pred.unnormed[,1]) ^ 2) / length(samples.test.ctr)

par(mfrow=c(1,2))
plot(samples.test.sold, pred.unnormed, col='red', main='samples vs predict', pch=18, cex=0.7)
abline(0, 1, lwd=2)


?rbf
library(RSNNS)
inputs <- as.matrix(seq(0,10,0.1))
outputs <- as.matrix(sin(inputs) + runif(inputs*0.2))
outputs <- normalizeData(outputs, "0_1")
model <- rbf(inputs, outputs, size=40, maxit=10000,
             initFuncParams=c(0, 1, 0, 0.01, 0.01),
             learnFuncParams=c(1e-8, 0, 1e-8, 0.1, 0.8),
             linOut=TRUE)
par(mfrow=c(2,1))
plotIterativeError(model)
plot(inputs, outputs)
lines(inputs, fitted(model), col="green")
