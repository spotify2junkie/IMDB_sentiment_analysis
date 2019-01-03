# Overview 

* Background:In this project, we use movie review comments to predict corresponding comments.
* Problem Statement: This project is about classfication task, key problem is about extracing information from every reviews and build numerical components out of text.
* Objective: Based on evaluation page, our objective is to get the probability of being positive commented, then evaluate all the test data predicted probability by using auc-metric.
* Input of the model: DTM(document-term matrix) is being used here as the input of the data, however the column number of DTM is 2000, and only 2000 words was being chosen for training models on.
* Output: Probability as mentioned.

# Customized vocabulary

* We extract vocabulary through t-test screening, this is based on normal distribution assumption about vocabulary being used.In general,after we transformed train data to DTM, we calculate the mean and variance for every terms(a single word or 2 words) and check their t-statistics and rank them based on absolute value. Check code below:
```{r,eval=FALSE}
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
```

# Techinical Details
* Model Used: Only used Ridge model to train the data, min of lambda was chosen by 10 fold cv.
* Data Preprocessing:
  - Stop words: Using stop words provided by professsor Liang
  - ngram: Choose ngram range (1,2).
  - frequent-infrequent pruning: removed terms that appears 50% in every comments and removed terms that only show up 5 times.
  - HTML Tag : html tags being removed and also all words being setted as lower case.
  
# Model Validation:
* Running Time: 3.601397 mins.
* Model Performance
```{r}
#create a table 
split=c("Performance for Split 1","Performance for Split 2",
"Performance for Split3","Vocabulary Size")
perf = c(0.9623,0.9623,0.9629,2000)
result4 = data.frame(split,perf)

colnames(result4) = c("1","2")

kable_styling(kable(result4, format = "latex", digits = 3), full_width = F) 
```

* Model Limitation: can only be applied to similiar text data, cannot be applied to new data that looks quiet different to old comments unless re-train the model 
* Future improvement: Can try some other models like randomeforest and further tune the DTM

# Rshiny Visulization

# [please click here](https://yuhuiluo.shinyapps.io/shiny/)



<img width="1346" alt="pic" src="https://user-images.githubusercontent.com/26384346/50667766-89953b80-0f80-11e9-8bb5-7f8b5036cbe4.png">



