#!/usr/bin/env Rscript

library(RSNNS)
source('../lib/utils.R')

data <- read.table('../../data/57_0811-0821-sold-tbsold-rate.ssv', header=TRUE)
res <- sail.sample(data, 0.9)
train_data <- res$main
test_data <- res$remain

names_ <- names(train_data)
formula_ <- paste(c(names_[1], paste(names_[2:length(names_)], collapse = " + ")), collapse=' ~ ')
f <- as.formula(formula_)

train_data_scaled <- sail.scale(train_data)

nn <- rbf(train_data_scaled, train_data_scaled$sold, hidden=c(3,2), linear.output=TRUE)
plot(nn)
