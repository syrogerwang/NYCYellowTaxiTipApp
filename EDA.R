library(arrow)
library(dplyr)
library(purrr)
library(tidyverse)
library(lubridate)
library(caret)

df <- read_parquet("yellow_tripdata_2025-02.parquet", header=TRUE)
names(df)
dim(df)
summary(df)

# Create tip_pct to better evaluate tipping as a percent of fare
df <- df %>% mutate(tip_pct = tip_amount / fare_amount * 100) %>%
  filter(payment_type %in% c(0,1), #only flex fare and credit cards
         fare_amount > 0,   #non-zero fares only
         trip_distance > 0, #non-zero distance only
         RatecodeID != 99,  #null/unknown values discarded
         tip_pct <= 100)    #limit to reasonable tip % only

# Join taxi pickup/drop off location IDs to taxi zone lookup information
locations <- read.csv("taxi_zone_lookup.csv", header = TRUE)

df <- df %>%
  left_join(locations, by = c("PULocationID" = "LocationID")) %>%
  left_join(locations, by = c("DOLocationID" = "LocationID"), 
            suffix = c("_pickup", "_dropoff"))

# verify join worked as intended
print(head(df), width=Inf)
dim(df)
summary(df)
dim(df[df$tip_pct==0,])[1]/dim(df)[1]*100
dim(df[df$tip_amount==10,])[1]/dim(df)[1]*100

# draw 1% sample from each RatecodeID; RatecodeID==1 dominates dataset
sample_df <- taxi %>% 
  group_by(Month) %>%
  sample_frac(0.05, replace=FALSE) %>%
  ungroup()

# create time features to explore patterns
df1 <- sample_prop %>% mutate(
  hour = lubridate::hour(tpep_pickup_datetime),
  timeofday = factor(ifelse(lubridate::hour(tpep_pickup_datetime) < 5, "EarlyAM",
              ifelse(lubridate::hour(tpep_pickup_datetime) < 10, "AM Commute",
              ifelse(lubridate::hour(tpep_pickup_datetime) < 14, "Lunch",
              ifelse(lubridate::hour(tpep_pickup_datetime) < 19, "PM Commute",
                     "Night"))))),
  weekday = lubridate::wday(tpep_pickup_datetime, label = TRUE),
  lunchtime = ifelse(10 < lubridate::hour(tpep_pickup_datetime) & 
                       lubridate::hour(tpep_pickup_datetime) < 14,1,0))
summary(df1)
names(df1)
# Pairs
pairs(~ trip_distance + RatecodeID + sqrt(fare_amount) + log(tip_amount) + tip_pct, data= df1)

dim(df1[df1$tip_amount==10,])[1]/dim(df1)[1]*100
model <- lm(tip_amount~trip_distance+fare_amount,data=df1)
summary(model)

model2 <- lm(tip_amount~trip_distance+fare_amount+I(RatecodeID^2),data=df1)
summary(model2)

# plot distribution of tips 
df %>%
  ggplot(aes(tip_amount)) + geom_histogram(bins=50)

df1 %>%
  ggplot(aes(tip_pct,fill=timeofday)) + geom_histogram(bins=50)

df1 %>%
  group_by(hour) %>%
  summarize(mean_tip = mean(tip_pct)) %>%
  ggplot(aes(mean_tip,fill=factor(hour))) + geom_histogram()

df1 %>% ggplot(aes(timeofday,fill = tip_pct>35)) + geom_bar(position="fill")

df1 %>% group_by(PULocationID, Zone_pickup, Borough_pickup) %>%
  summarize(mean_tip = mean(tip_pct)) %>%
  arrange(desc(mean_tip))

#Rate code(1-standard,2-JKF,3-Newark,4-Nassau/Westchester,5-Negotiated,6-Group ride)
df1 %>% group_by(RatecodeID) %>% 
  summarize(mean_tip = mean(tip_pct)) %>%
  ggplot(aes(RatecodeID,mean_tip)) + geom_bar(stat = "identity")

# Scatterplots
df1 %>% ggplot(aes(trip_distance, tip_pct)) +
  geom_point()

df1 %>% ggplot(aes(trip_distance, tip_amount)) +
  geom_point() + geom_smooth(method="lm")

df1 %>% ggplot(aes(Borough_pickup, tip_amount)) +
  geom_point(aes(col=Borough_dropoff),position="jitter")

# Correlations
df1 %>%
  select(tip_pct, tip_amount, RatecodeID, trip_distance, fare_amount, hour) %>%
  cor(use = "complete.obs")

# Training/Test split
train_ind <- sample(1:dim(df1)[1],round(.7*dim(df1)[1]))
train = df1[train_ind,]
test = df1[-train_ind,]
# Linear Model
fit <- lm(tip_amount~trip_distance+I(trip_distance^.5)+I(fare_amount^.5)+RatecodeID, data= train)
summary(fit)

pred_probs <- predict(fit,test)
RMSE <- sqrt(mean((pred_probs-test$tip_amount)^2))
MAE <- mean(abs(pred_probs-test$tip_amount))

RMSE
MAE
