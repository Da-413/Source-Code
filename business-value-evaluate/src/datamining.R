dat <- read.csv("C:/Users/gyoo4/OneDrive/바탕 화면/데이터마이닝/data_re.csv",
                header = T, fileEncoding = "euc-kr")

dat1 <- dat[,-1]
rownames(dat1) <- dat[,1]

dat2 <- apply(dat1, 2, scale)
dat2 <- as.data.frame(dat2)
colnames(dat2) <- colnames(dat1)
rownames(dat2) <- dat[,1]



model <- lm(시가총액 ~ ., data = dat2)
summary(model)

step(model, direction='both')
formula = '시가총액 ~ RIM기업가치 + 매출액증가율 + ROA + ROI + 자기자본비율'
best.model <- lm(formula, data = dat2)
summary(best.model)

train_dat2 <- dat2[n1,]; test_dat2 <- dat2[-n1,]; train_y2 <- dat2[n1,1]; test_y2 <- dat2[-n1,1]
best.model <- lm(formula, data = train_dat2)
summary(best.model)
mean((test_y2-predict(best.model,test_dat2))^2)


install.packages("glmnet")
library(glmnet)

model.BIC = 5000
x1 = x2 = x3 = x4 = x5 = 0
for(a in 1:4){
  for(b in 1:4){
    for(c in 1:4){
      for(d in 1:4){
        for(e in 1:4){
          model = lm(시가총액 ~  poly(RIM기업가치,a) + poly(매출액증가율, b)
                     + poly(자기자본비율, c) + poly(ROA, d) + poly(ROI, e), data = dat2)
          if(BIC(model) <= model.BIC){
            model.BIC <- BIC(model)
            x1=a; x2=b; x3=c; x4=d; x5=e
          }
        }
      }
    }
  }
}
x1; x2; x3; x4; x5
poly.model = lm(시가총액 ~ poly(RIM기업가치,4) + poly(매출액증가율, 4) + poly(자기자본비율, 2)
                + poly(ROA, 4) + ROI + 총자산회전율, data = dat2)
summary(poly.model)

par(mfrow=c(2,2))

plot(poly.model)

library(outliers)   
library(DMwR)       
library(tsoutliers) 
library(car)
outlierTest(best.model)
cooks <- cooks.distance(best.model)
cooks[cooks > 4/nrow(dat2)]
outlierTest(poly.model)
cooks <- cooks.distance(poly.model); cooks[cooks > 4/nrow(dat2)]



#span값을 구하기 위한 5-fold cross validation
K = 5
idx = sample(rep(1:K, length = nrow(dat3)))
span = c(0.3,0.4,0.5,0.6,0.7,0.8,0.9)
loess_mse <- matrix(0, nrow=length(span), ncol=K+1)

for(j in 1:length(span)){
  
  for(i in 1:K){
    
    train = dat2[which(idx != i),]
    test = dat2[which(idx == i),]
    
    local_regression = loess(시가총액 ~ RIM기업가치 + 매출액증가율 + 자기자본비율 + ROI,
                             span = span[j], data = train)
    pred = predict(local_regression, newdata = test)
    loess_mse[j,i] = mean((pred - test$시가총액)^2, na.rm = T)
    
  }
  
  loess_mse[j,6] = sum(loess_mse[j,1],loess_mse[j,2],loess_mse[j,3],loess_mse[j,4],loess_mse[j,5]) / 5
  
} #span = 1이 mse가 가장 낮음
loess_mse

loess_model1 = loess(시가총액 ~ RIM기업가치 + 매출액증가율 + 자기자본비율 + ROI,
                     span = 0.6, data = dat2)
summary(loess_model1) #dat3을 이용한 loess model, loess모형은 변수가 최대 4개 밖에 안들어감

new_data = read.csv("C:/Users/gyoo4/OneDrive/바탕 화면/데이터마이닝/new_data.csv",
                    header = T, fileEncoding = "euc-kr")

new_data2 = new_data[,-1]
rownames(new_data2) <- new_data[,1]

new_data1 = apply(new_data2, 2, scale)
new_data1 = as.data.frame(new_data1)
colnames(new_data1) <- colnames(new_data2)
rownames(new_data1) <- new_data[,1]
new_data = new_data1

pred_loess = predict(loess_model1, new_data)
sum(is.na(pred_loess))

pred_poly = predict(poly.model, new_data)
p = predict(poly.model, new_data, interval='prediction')
c = predict(poly.model, new_data, interval='confidence')



library(Metrics)

시가총액 <- dat3$시가총액
RIM기업가치 <- dat3$RIM기업가치
매출액증가율 <- dat3$매출액증가율
자기자본비율 <- dat3$자기자본비율
매출총이익률 <- dat3$매출총이익률
ROA <- dat3$ROA
ROI <- dat3$ROI
ss.model1 <- smooth.spline(시가총액 ~ RIM기업가치 + 매출액증가율 + 자기자본비율 + 매출총이익률 + ROA + ROI,
                           cv = TRUE)
ss.model1$df
ss.model2 <- smooth.spline(시가총액 ~ RIM기업가치 + 매출액증가율 + 자기자본비율 + 매출총이익률 + ROA + ROI,
                           df=16)

summary(ss.model2)
ss.model2



rmse <- matrix(0, nrow=100, ncol=3)
mape <- matrix(0, nrow=100, ncol=3)

for(i in 1:100){
  
  n1<-sample(1:nrow(dat2), 100)
  
  train_dat2 <- dat2[n1,]; test_dat2 <- dat2[-n1,]; train_y2 <- dat2[n1,1]; test_y2 <- dat2[-n1,1]
  
  best.model <- lm(시가총액 ~ RIM기업가치 + 매출액증가율 + 자기자본비율 + 매출총이익률 + ROA + ROI,
                   data = train_dat2)
  pred.best.model <- predict(best.model, newdata = test_dat2)
  rmse.best.model = sqrt(mse(test_y2, pred.best.model))
  mape.best.model = mape(test_y2, pred.best.model)
  
  poly.model = lm(시가총액 ~ poly(RIM기업가치,4) + poly(매출액증가율, 4) + poly(자기자본비율, 2)
                  + poly(ROA, 4) + ROI + 총자산회전율, data = train_dat2)
  pred.poly.model <- predict(poly.model, newdata = test_dat2)
  rmse.poly.model = sqrt(mse(test_y2, pred.poly.model))
  mape.poly.model = mape(test_y2, pred.poly.model)
  
  loess_model = loess(시가총액 ~ RIM기업가치 + 매출액증가율 + 자기자본비율 + ROI,
                      span = 0.6, data = train_dat2)
  pred.loess.model <- predict(loess_model, newdata = test_dat2)
  rmse.loess.model = sqrt(mean((test_y2-pred.loess.model)^2, na.rm=T))
  mape.loess.model = mean(abs((test_y2-pred.loess.model)/test_y2), na.rm=T)
  
  
  
  rmse[i,] <- c(rmse.best.model, rmse.poly.model, rmse.loess.model)
  mape[i,] <- c(mape.best.model, mape.poly.model, mape.loess.model)
}

rmse
mape

mean.rmse = apply(rmse, 2, mean)
mean.mape = apply(mape, 2, mean)

mean.rmse = matrix(mean.rmse, nrow=1, byrow=T)

ind<-matrix(c(mean.rmse, mean.mape),nrow=2, byrow=T)
rownames(ind) <- c("mean_rmse","mean_mape")
colnames(ind) <- c("best_model2", "poly_model2", "loess_model2", "best_model3", "poly_model3", "loess_model3", "best_poly_model3", "poly_model3")

ind





par(mfrow = c(1,1))
plot(dat3$시가총액)


x = model.matrix(시가총액~.,dat2)[,-1]
y = dat2$시가총액

set.seed(1)
train = sample(1:nrow(x),nrow(x)/2)
test = (-train)
y.test = y[test]
cv.out=cv.glmnet(x[train,],y[train],alpha=0)
par(mfrow=c(1,1))
plot(cv.out)
bestlam=cv.out$lambda.min
bestlam

ridge.mod=glmnet(x[train,],y[train],alpha=0,lambda=bestlam)
ridge.pred=predict(ridge.mod,s=bestlam,newx=x[test,])
summary(ridge.mod)
mean((ridge.pred-y.test)^2)




cv.out1=cv.glmnet(x[train,],y[train],alpha=1)
plot(cv.out1)
bestlam=cv.out1$lambda.min
bestlam
lasso.mod=glmnet(x[train,],y[train],alpha=1,lambda=bestlam)
lasso.pred=predict(lasso.mod,s=bestlam,newx=x[test,])
summary(lasso.mod)
mean((lasso.pred-y.test)^2)

out=glmnet(x,y,alpha=1)
lasso.coef=predict(out,type="coefficients",s=bestlam)[1:11,]
lasso.coef
lasso.coef[lasso.coef!=0]




