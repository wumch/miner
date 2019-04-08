
library(rpart)
library(rpart.plot)
library(pROC)
library(RMySQL)
set.seed(1)

# db_con <- dbConnect(MySQL(), host="59805caa02035.gz.cdb.myqcloud.com", port=5339, dbname="evaluate", user="memuu_dev", password="memuu-dev@2101")
db_con <- dbConnect(MySQL(), host="localhost", dbname="evaluate", user="root", password="root")
db_resh <- dbSendQuery(db_con, "
select c.read_num as `read_score`, 
    if(c.has_video, 'yes', 'no') as has_video,
    ifnull(char_length(c.content), 0) as chars, ifnull(char_length(c.content) / cg.avg_chars, 0) as `chars_score`,
    if(c.content is not null and c.content!='', 'yes', 'no') as has_content,
    c.pic_num, ifnull(c.pic_num / cg.avg_pic_num, 0) as `pic_score`,
    if(c.pic_num>0, 'yes', 'no') as has_pic,
    ifnull(char_length(c.append_content), 0) as a_chars, ifnull(char_length(c.append_content) / cg.avg_append_chars, 0) as `a_chars_score`,
    if(c.append_content is not null and c.append_content!='', 'yes', 'no') as has_a_content,
    c.append_pic_num, ifnull(c.append_pic_num / cg.avg_append_pic_num, 0) as `a_pics_score`,
    if(c.append_pic_num is not null and c.append_pic_num>0, 'yes', 'no') as has_append_pic
from eva_taobao_comment c
left join eva_taobao_comment_group cg on cg.tbid=c.tbid
where c.read_num>0
    and c.date between '2017-10-01' and '2017-10-31' 
    and c.tbid=555786640442
order by rand() limit 100000
")
comments <- dbFetch(db_resh, n=-1)
dbClearResult(db_resh)
dbDisconnect(db_con)


train.index <- sample(1:nrow(comments), round(nrow(comments) * 0.8))
comments.train <- comments[train.index, ]
comments.test <- comments[-train.index, ]

comments.train.mean <- mean(comments.train$read_score)
comments.train.mse <- sum(sqrt((comments.train$read_score - comments.train.mean) ** 2)) / length(comments.train$read_score)
comments.test.mean <- mean(comments.test$read_score)
comments.test.mse <- sum(sqrt((comments.test$read_score - comments.test.mean) ** 2)) / length(comments.test$read_score)


model <- rpart(read_score ~ ., data=comments.train, method="anova")
rpart.plot(model, type=1, digits=4, main="评价阅读数(未剪枝)")
pred.model <- predict(model, newdata=comments.test)
pred.model.mse <- sum(sqrt((pred.model - comments.test.mean) ** 2)) / length(pred.model)


cp <- model$cptable[which.min(model$cptable[,"xerror"]),"CP"]
model.cut <- prune(model, cp=0.001)
rpart.plot(model.cut, type=1, digits=4, main="评价阅读数(已剪枝)")
pred.model.cut <- predict(model.cut, newdata=comments.test)
pred.model.cut.mse <- sum(sqrt((pred.model.cut - comments.test.mean) ** 2)) / length(pred.model)

