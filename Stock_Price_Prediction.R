library(quantmod)
library(tidyverse)
library(dplyr)
library(ggplot2)
library(caret)
library(randomForest)
library(e1071)
library(tidymodels)
#Retreiving Stock Price using quantmod

library(quantmod)
library(xts)
library(ggplot2)
library(gridExtra) # grid.arrange

graphics.off()
rm(list=ls())


# Retrieve Samsung data 
sam_data <- getSymbols(Symbols = "005930.KS", src = "yahoo", from = Sys.Date() - 5000, 
                        to = Sys.Date(), auto.assign = FALSE)

View(sam_data)
#We only need to select relevant features for analysis. Here, I intend to discard irrelevant attributes 
#that do not significantly impact the stock market and focus on key attributes that affect the stock market, such as Timestamp, Open, High, Low, Close, Volume, and Weighted Price. By doing so, we can concentrate on analyzing factors closely related to stock market behavior and price fluctuations. This enables us to gain deeper insights into market behavior, volatility, 
#and potential driving factors, providing valuable support for predicting future stock price trends.

#remove missing data
sam_data <- na.omit(sam_data)

# Convert xts object to dataframe
df_data <- as.data.frame(sam_data)

# Calculate the difference in closing prices
df_data$Price_Diff <- c(NA, diff(df_data$`005930.KS.Close`))

# Create a new column indicating the direction
df_data$Direction <- ifelse(df_data$Price_Diff > 0, "Up", "Down")
                            

sam_data <- na.omit(df_data)
sam_data <- sam_data[, -7]

# Convert dataframe to xts
sam_xts <- xts(sam_data[, -ncol(sam_data)], order.by = as.Date(rownames(sam_data)))
sam_data$Direction <- as.factor(sam_data$Direction)
# Display the converted xts object
print(sam_xts)

#EDA
chart_Series(sam_xts, col = "black")
add_SMA(n = 100, on = 1, col = "red")
add_SMA(n = 20, on = 1, col = "black")
add_RSI(n = 14, maType = "SMA")
add_MACD(fast = 12, slow = 25, signal = 9, maType = "SMA", histogram = TRUE)


barplot(table(sam_data$Direction), col = "blue", main = "Stock Direction")
library(Hmisc)
sam_log <- log(sam_xts)
head(sam_log, n = 10)

library(tidyverse)
glimpse(sam_xts)
describe(sam_xts)

# Splitting the dataset into training and testing sets
set.seed(123) # for reproducibility
data_split <- initial_split(sam_data,
                             prop = 0.75,
                             strata = Direction)


# Extract the training and testing set
train_data <- training(data_split)
test_data <- testing(data_split)
# Create preprocessing recipe
sam_rec <- recipe(Direction ~ ., data =train_data) %>% 
  # Normalize numeric variables
  step_normalize(all_numeric_predictors())


# Print summary recipe
print(summary(sam_rec))

# Create a logistic regression model specification
lg_spec <- 
  # Type of model
  logistic_reg() %>% 
  # Engine
  set_engine("glm") %>% 
  # Mode
  set_mode("classification")

# Boosted tree regression model specification
boost_spec <- boost_tree() %>% 
  set_engine("xgboost") %>% 
  set_mode("classification")

#Random forest
rd_spec <- 
  # Type of model
  rand_forest() %>% 
  # Engine
  set_engine("ranger") %>% 
  # Mode
  set_mode("classification")

# Logistic regression workflow
lg_wf <- workflow() %>% 
  add_recipe(sam_rec) %>% 
  add_model(lg_spec)

lg_wf


# xgboost workflow
boost_wf <- workflow() %>% 
  add_recipe(sam_rec) %>% 
  add_model(boost_spec)

boost_wf

# random forest
rd_wf <- workflow() %>% 
  add_recipe(sam_rec) %>% 
  add_model(rd_spec)

rd_wf

set.seed(123)
sam_boot <- bootstraps(data = train_data)
head(sam_boot)
unique(sam_boot$id)

# Fit boosted trees model to the resamples
boost_rs <- fit_resamples(
  object = boost_wf,
  resamples = sam_boot,
  metrics = metric_set(accuracy)
)

# Show the model with best metrics
show_best(boost_rs) %>% 
  as.data.frame()

#logistic regression
lg_rs <- fit_resamples(
  object = lg_wf,
  resamples = sam_boot,
  metrics = metric_set(accuracy)
)

# Show the model with best metrics
show_best(lg_rs) %>% 
  as.data.frame()


#For rd
rd_rs <- fit_resamples(
  object = rd_wf,
  resamples = sam_boot,
  metrics = metric_set(accuracy)
)

# Show the model with best metrics
show_best(rd_rs) %>% 
  as.data.frame()

# Finalize the workflow
final_wf <- rd_wf %>% 
  finalize_workflow(parameters = select_best(lg_rs))

final_wf
# Make a last fit
final_fit <- final_wf %>% 
  last_fit(data_split)

# Collect metrics
final_fit %>% 
  collect_metrics() %>% 
  as.data.frame()

# Create confusion matrix
collect_predictions(final_fit) %>% 
  conf_mat(truth = Direction, estimate = .pred_class)

# Visualize confusion matrix
collect_predictions(final_fit) %>% 
  conf_mat(truth = Direction, estimate = .pred_class) %>% 
  autoplot(type = "heatmap")

# Other metrics that arise from confusion matrix
collect_predictions(final_fit) %>% 
  conf_mat(truth = Direction, estimate = .pred_class) %>% 
  summary() %>% 
  filter(.metric %in% c("accuracy", "sens", "ppv", "f_meas")) %>% 
  as.data.frame()

#Regression
models <- lm(`005930.KS.Open` ~ `005930.KS.Adjusted`, data = sam_data)
ggplot(sam_data, aes(x = `005930.KS.Adjusted`, y =`005930.KS.Open` )) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  labs(x = "Stock.Adjusted", y = "Stock.Open", title = "Linear Regression")


cor_matrix <- cor(sam_data[,-7])
cor_data <- reshape2::melt(cor_matrix)

ggplot(cor_data, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient(low = "blue", high = "red") +
  labs(title = "Correlation Matrix Heatmap")


library(caTools)
library(forecast)
sam_log = sam_log[, -7]
train_data <- sam_log[1:2800, "005930.KS.Close"]  
set.seed(123)
arima_model <- auto.arima(train_data, stationary = TRUE, ic = c("aicc", "aic", "bic"), 
                          trace = TRUE)
summary(arima_model)
checkresiduals(arima_model)

arima <- arima(train_data, order = c(0, 0, 5))
summary(arima)

close_prices <- Cl(sam_data)
forecast1 <- forecast(arima, h = 100)
plot(forecast1)

train_datas <- sam_log[1:2800, "005930.KS.Close"]
arima <- arima(train_datas, order = c(0, 0, 5))
forecast_ori <- forecast(arima, h = 100)
a <- ts(train_datas)
forecast_ori %>% autoplot() + autolayer(a)

str(sam_data)
summary(sam_data)

arima <- arima(train_data, order = c(0, 0, 5))
forecast1 <- forecast(arima, h = 573)
forecasted_values <- forecast1$mean
actual_values <- coredata(sam_log[2801:3373, "005930.KS.Close"])
errors <- actual_values - forecasted_values
mse <- mean(errors^2)
rmse <- sqrt(mse)  
mae <- mean(abs(errors))
mape <- mean(abs(errors/actual_values)) * 100

cat("Mean Squared Error (MSE):", mse, "\n")
cat("Root Mean Squared Error (RMSE):", rmse, "\n")
cat("Mean Absolute Error (MAE):", mae, "\n")
cat("Mean Absolute Percentage Error (MAPE):", mape, "%\n")


