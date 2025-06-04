#dat2 : 표준화
dat <- read.csv("C:/Users/gyoo4/OneDrive/바탕 화면/데이터마이닝/data_re.csv",
                header = T, fileEncoding = "euc-kr")

dat1 <- dat[,-1]
rownames(dat1) <- dat[,1]

dat2 <- apply(dat1, 2, scale)
dat2 <- as.data.frame(dat2)
colnames(dat2) <- colnames(dat1)
rownames(dat2) <- dat[,1]

model <- lm(시가총액 ~ ., data = dat2) #linear model
summary(model)

step(model, direction='both')
best.model <- lm(시가총액 ~ RIM기업가치 + 유동비율 + ROA + 부채비율,
                 data = dat2)
summary(best.model) #best subset selection model

library(glmnet)

x1; x2; x3; x4; x5; x6; x7
poly.model = lm(시가총액 ~ poly(RIM기업가치,4) + poly(매출액증가율, 4) + 부채비율 + 매출총이익률 
                + ROA + ROI + 총자산회전율, data = dat2)
summary(poly.model) #4차까지 polynomial model 중 BIC가 가장 낮은 모형


#dat3 : 삼성바이오로직스, 삼성SDI 제거
which(dat2[,1]>=1.234867)
dat2[119,]
dat3 <- dat2[-c(23,119),] #삼성바이오로직스, 삼성SDI


model1 <- lm(시가총액 ~ ., data = dat3)
summary(model1) #dat3으로 적합한 linear model

step(model1, direction='both')
best.model1 <- lm(시가총액 ~ RIM기업가치 + 매출액증가율 + 부채비율 + 매출총이익률 + ROA + ROI + 총자산회전율,
                  data = dat3)
summary(best.model1) #dat3에서 best subset selection

poly.model1 = lm(시가총액 ~ poly(RIM기업가치,4) + poly(매출액증가율, 4) + 부채비율 + poly(매출총이익률,2) 
                 + ROA + ROI + 총자산회전율, data = dat3)
summary(poly.model1) #dat3에서 polynomial 중 BIC가 가장 낮은 모형

#span값을 구하기 위한 5-fold cross validation
K = 5
idx = sample(rep(1:K, length = nrow(dat3)))
span = c(0.3,0.4,0.5,0.6,0.7,0.8,0.9)
loess_mse <- matrix(0, nrow=7, ncol=6)

for(j in 1:length(span)){
  
  for(i in 1:K){
    
    train = dat3[which(idx != i),]
    test = dat3[which(idx == i),]
    
    local_regression = loess(시가총액 ~ RIM기업가치 + 매출액증가율 + 부채비율 + 총자산회전율,
                             span = span[j], data = train)
    pred = predict(local_regression, newdata = test)
    loess_mse[j,i] = mean((pred - test$시가총액)^2, na.rm = T)
    
  }
  
  loess_mse[j,6] = mean(loess_mse[j,1],loess_mse[j,2],loess_mse[j,3],loess_mse[j,4],loess_mse[j,5])
  
} #span = 0.8이 mse가 가장 낮음

loess_model1 = loess(시가총액 ~ RIM기업가치 + 매출액증가율 + 부채비율 + 총자산회전율,
                     span = 0.8, data = dat3)
summary(loess_model1) #dat3을 이용한 loess model, loess모형은 변수가 최대 4개 밖에 안들어감


#dat2와 dat3을 각각 이용한 best selection model, polynomial model, local regression model
#총 6개 model의 rmse와 mape(오차제곱항에 절댓값을 이용한 지표) 비교

library(Metrics)

rmse <- matrix(0, nrow=100, ncol=6)
mape <- matrix(0, nrow=100, ncol=6)

for(i in 1:100){
  
  n1<-sample(1:nrow(dat2), 100)
  n2<-sample(1:nrow(dat3), 100)
  
  train_dat2 <- dat2[n1,]; test_dat2 <- dat2[-n1,]; train_y2 <- dat2[n1,1]; test_y2 <- dat2[-n1,1]
  train_dat3 <- dat3[n2,]; test_dat3 <- dat3[-n2,]; train_y3 <- dat3[n2,1]; test_y3 <- dat3[-n2,1]
  
  best.model <- lm(시가총액 ~ RIM기업가치 + 매출액증가율 + 부채비율 + 매출총이익률 + ROA + ROI + 총자산회전율,
                   data = train_dat2)
  pred.best.model <- predict(best.model, newdata = test_dat2)
  rmse.best.model = sqrt(mse(test_y2, pred.best.model))
  mape.best.model = mape(test_y2, pred.best.model)
  
  poly.model = lm(시가총액 ~ poly(RIM기업가치,4) + poly(매출액증가율, 4) + 부채비율 + 매출총이익률 
                  + ROA + ROI + 총자산회전율, data = train_dat2)
  pred.poly.model <- predict(poly.model, newdata = test_dat2)
  rmse.poly.model = sqrt(mse(test_y2, pred.poly.model))
  mape.poly.model = mape(test_y2, pred.poly.model)
  
  loess_model = loess(시가총액 ~ RIM기업가치 + 매출액증가율 + 부채비율 + 총자산회전율,
                      span = 0.8, data = train_dat2)
  pred.loess.model <- predict(loess_model, newdata = test_dat2)
  rmse.loess.model = sqrt(mean((test_y2-pred.loess.model)^2, na.rm=T))
  mape.loess.model = mean(abs((test_y2-pred.loess.model)/test_y2), na.rm=T)
  
  
  
  best.model1 <- lm(시가총액 ~ RIM기업가치 + 매출액증가율 + 부채비율 + 매출총이익률 + ROA + ROI + 총자산회전율,
                    data = train_dat3)
  pred.best.model1 <- predict(best.model1, newdata = test_dat3)
  rmse.best.model1 = sqrt(mse(test_y3, pred.best.model1))
  mape.best.model1 = mape(test_y3, pred.best.model1)
  
  poly.model1 = lm(시가총액 ~ poly(RIM기업가치,4) + poly(매출액증가율, 4) + 부채비율 + poly(매출총이익률,2) 
                   + ROA + ROI + 총자산회전율, data = train_dat3)
  pred.poly.model1 <- predict(poly.model1, newdata = test_dat3)
  rmse.poly.model1 = sqrt(mse(test_y3, pred.poly.model1))
  mape.poly.model1 = mape(test_y3, pred.poly.model1)
  
  loess_model1 = loess(시가총액 ~ RIM기업가치 + 매출액증가율 + 부채비율 + 총자산회전율,
                       span = 0.8, data = train_dat3)
  pred.loess.model1 <- predict(loess_model1, newdata = test_dat3)
  rmse.loess.model1 = sqrt(mean((test_y3-pred.loess.model1)^2, na.rm=T))
  mape.loess.model1 = mean(abs((test_y3-pred.loess.model1)/test_y3), na.rm=T)
  
  
  rmse[i,] <- c(rmse.best.model, rmse.poly.model, rmse.loess.model, rmse.best.model1,
                rmse.poly.model1, rmse.loess.model1)
  mape[i,] <- c(mape.best.model, mape.poly.model, mape.loess.model, mape.best.model1,
                mape.poly.model1, mape.loess.model1)
}

rmse
mape

mean.rmse = apply(rmse, 2, mean)
mean.mape = apply(mape, 2, mean)

ind<-matrix(c(mean.rmse, mean.mape),nrow=2, byrow=T)
rownames(ind) <- c("mean_rmse","mean_mape")
colnames(ind) <- c("best_model2", "poly_model2", "loess_model2", "best_model3", "poly_model3", "loess_model3")

#결과는 polynomial은 지표가 높게 나타나고 local regression보다는 linear regression이 약간 좋은 성능을 보임.


#smooth.spline은 predict결과가 다르게 지표를 계산하지 못함
시가총액 <- dat3$시가총액
RIM기업가치 <- dat3$RIM기업가치
매출액증가율 <- dat3$매출액증가율
부채비율 <- dat3$부채비율
매출총이익률 <- dat3$매출총이익률
ROA <- dat3$ROA
ROI <- dat3$ROI
총자산회전율 <- dat3$총자산회전율
ss.model1 <- smooth.spline(시가총액 ~ RIM기업가치 + 매출액증가율 + 부채비율 + 매출총이익률,
                           cv = TRUE)
ss.model1$df
ss.model2 <- smooth.spline(시가총액 ~ RIM기업가치 + 매출액증가율 + 부채비율 + 매출총이익률,
                           cv = TRUE, df=16)

summary(ss.model1)