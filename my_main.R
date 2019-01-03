# install.packages("text2vec")
# install.packages("data.table")
# install.packages("magrittr")
# install.packages("tm")
start_time <- Sys.time()

end_time <- Sys.time()

library(text2vec)
library(data.table)
library(magrittr)
library(glmnet)
library(curl)
library(tm)
library(pROC)
all = read.table("data.tsv",stringsAsFactors = F,header = T)
all$review = gsub('<.*?>', ' ', all$review)
splits = read.table("splits.csv", header = T)
s = 1  # Here we get the 3rd training/test split. 
train = all[-which(all$new_id%in%splits[,s]),]
test = all[which(all$new_id%in%splits[,s]),]

prep_fun = tolower
tok_fun = word_tokenizer

stop_words = c("i", "me", "my", "myself", 
               "we", "our", "ours", "ourselves", 
               "you", "your", "yours", 
               "their", "they", "his", "her", 
               "she", "he", "a", "an", "and",
               "is", "was", "are", "were", 
               "him", "himself", "has", "have", 
               "it", "its", "of", "one", "for", 
               "the", "us", "this")

it_train = itoken(train$review,preprocessor = prep_fun,tokenizer = tok_fun, 
                  ids = train$new_id,progressbar = FALSE)
it_test=itoken(test$review,preprocessor = prep_fun,tokenizer = tok_fun, 
               ids = test$new_id,progressbar = FALSE)
vocab = create_vocabulary(it_train,ngram = c(ngram_min = 1L, ngram_max
                                             = 1L), 
                          stopwords = stop_words)
pruned_vocab = prune_vocabulary(vocab,
                                term_count_min = 5, 
                                doc_proportion_max = 0.5,
                                doc_proportion_min = 0.001)
vectorizer = vocab_vectorizer(pruned_vocab)
dtm_train = create_dtm(it_train, vectorizer)
dtm_test = create_dtm(it_test, vectorizer)
# identical(rownames(dtm_train), train$new_id)
v.size = dim(dtm_train)[2]
ytrain = train$sentiment

summ = matrix(0, nrow=v.size, ncol=4)
summ[,1] = apply(dtm_train[ytrain==1, ], 2, mean)
summ[,2] = apply(dtm_train[ytrain==1, ], 2, var)
summ[,3] = apply(dtm_train[ytrain==0, ], 2, mean)
summ[,4] = apply(dtm_train[ytrain==0, ], 2, var)
n1=sum(ytrain); 
n=length(ytrain)
n0= n - n1

myp = (summ[,1] - summ[,3])/
  sqrt(summ[,2]/n1 + summ[,4]/n0)

words = colnames(dtm_train)
id = order(abs(myp), decreasing=TRUE)[1:2000]
pos.list = words[id[myp[id]>0]]
neg.list = words[id[myp[id]<0]]
pos.freq = myp[id[myp[id]>0]]
neg.freq = abs(myp[id[myp[id]<0]])


identical(length(pos.list),length(pos.freq)) # check whether length is same 
identical(length(neg.list),length(neg.freq)) # check whether length is same 

pos = data.frame(pos.list=pos.list,
                 pos.freq=pos.freq)
neg = data.frame(neg.list=neg.list,
                 neg.freq=neg.freq)
write.csv(pos,"pos.csv",row.names=FALSE)
write.csv(neg,"neg.csv",row.names=FALSE)

set.seed(500)
NFOLDS = 10
mycv = cv.glmnet(x=dtm_train[, id], y=train$sentiment, 
                 family='binomial',type.measure = "auc", 
                 nfolds = NFOLDS, alpha=0)
myfit = glmnet(x=dtm_train[, id], y=train$sentiment, 
               lambda = mycv$lambda.min, family='binomial', alpha=0)
logit_pred = predict(myfit, dtm_test[, id], type = "response")
roc_obj = roc(test$sentiment, as.vector(logit_pred))
auc(roc_obj) 
submission_to_save=data.table(new_id=test$new_id,
                              prob=logit_pred)
names(submission_to_save)[2] = "prob"
write.table(submission_to_save,"Result_1.txt",sep=",",row.names=FALSE)
end_time <- Sys.time()
end_time - start_time