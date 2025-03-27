# MAIN ANALYSIS ------------------------

# load the packages
library(caTools)
library(caret)
library(dplyr)
library(doParallel)
library(gbm)
library(ggplot2)
library(glmnet)
library(ipred)
library(keras)
library(tensorflow)
library(tinytex)
library(lubridate)
library(parallel)
library(randomForest)
library(ranger)
library(rBayesianOptimization)
library(readr)
library(stringr)
library(tensorflow)
library(xgboost)


# Final Data Preparation ----------
df_processed <- read.csv("Final_data.csv", stringsAsFactors = FALSE)

#Removing all remaining non-numerical variables and missing cells
df_processed <- df_processed %>% select(-c(combined_description, room_type))
df_processed <- na.omit(df_processed)
sum(is.na(df_processed))

# Table of variables for .RMD purposes
variable_table <- data.frame(
  Variable = names(df_processed),
  Type = sapply(df_processed, class)  # Get the data type of each column
)

variable_table <- data.frame(
  Variable = names(df_processed),
  Type = sapply(df_processed, function(x) paste(class(x), collapse = ", ")),
  Missing = sapply(df_processed, function(x) sum(is.na(x))),
  Unique = sapply(df_processed, function(x) length(unique(x))),
  stringsAsFactors = FALSE
)

# knitr::kable(variable_table, format = "latex", caption = "Variable Summary", booktabs = TRUE)

# The dataset is partially corrupted with overpriced listings (see Appendix)
# Therefore I remove top and bottom 1% of outliers:

# Calculate the 0.5th and 99.5th percentiles
percentile_low <- quantile(df_processed$price, 0.005, na.rm = TRUE)
percentile_high <- quantile(df_processed$price, 0.995, na.rm = TRUE)
cat("0.5th percentile of price:", percentile_low, "\n")
cat("99.5th percentile of price:", percentile_high, "\n")

# How many observations were removed - confirming the numbers for internal purposes
df_processed <- df_processed[df_processed$price >= percentile_low & df_processed$price <= percentile_high, ]
cat("Number of observations after removing outliers:", nrow(df_processed), "\n")

# Visualisation
Price_distribution <- ggplot(df_processed, aes(x = price)) +
  geom_histogram(bins = 30, fill = "lightgreen", color = "black") +
  ggtitle("Price Distribution After Filtering") +
  theme_minimal()
print(Price_distribution)

# I log the price variable (see Appendix)
df_processed$price <- as.numeric(df_processed$price)
df_processed$price <- log(df_processed$price)

# Improving the speed of calculations:
cores <- detectCores() - 1
cl <- makeCluster(cores)  
registerDoParallel(cl)    

# Splitting into training and validation sets:
set.seed(123)
split <- sample.split(df_processed$price, SplitRatio = 0.8)
training_set <- subset(df_processed, split == TRUE)
validation_set <- subset(df_processed, split == FALSE)
cat("Training set dimensions:", dim(training_set), "\n")
cat("Validation set dimensions:", dim(validation_set), "\n")

#Comparing the validation and training sets
summary_stats <- function(data, dataset_name) {
  data %>%
    summarise(
      Dataset = dataset_name,
      Mean = mean(price, na.rm = TRUE),
      Median = median(price, na.rm = TRUE),
      Variance = var(price, na.rm = TRUE),
      Std_Dev = sd(price, na.rm = TRUE),
      Min = min(price, na.rm = TRUE),
      Max = max(price, na.rm = TRUE),
      Count = n()
    )
}

train_stats <- summary_stats(training_set, "Training Set")
valid_stats <- summary_stats(validation_set, "Validation Set")
comparison_table <- bind_rows(train_stats, valid_stats)
print(comparison_table)

# Matrix Conversion for glmnet for L1, L2 and elastic:
x_train <- model.matrix(price ~ ., training_set)[, -1]
y_train <- training_set$price
x_valid <- model.matrix(price ~ ., validation_set)[, -1]
y_valid <- validation_set$price # exclude the 'price' !!!

# Lasso (L1) ------------------------
set.seed(123)
lasso_model <- glmnet(x_train, y_train, alpha = 1)  # alpha = 1 for Lasso

# Cross-validation to find 'optimal' lambda, testing on validation set
cv_lasso <- cv.glmnet(x_train, y_train, nfolds = 10, alpha = 1)
best_lambda_lasso <- cv_lasso$lambda.min
lasso_train_pred <- predict(lasso_model, s = best_lambda_lasso, newx = x_train)
lasso_valid_pred <- predict(cv_lasso, s = best_lambda_lasso, newx = x_valid)

# RMSE and R²
rmse_valid_lasso <- sqrt(mean((y_valid - lasso_valid_pred)^2))
rmse_train_lasso <- sqrt(mean((y_train - lasso_train_pred)^2))
ss_total <- sum((validation_set$price - mean(validation_set$price))^2)  # TSS
lasso_ss_residual <- sum((validation_set$price - lasso_valid_pred)^2)   # RSS
r_squared_lasso <- 1 - (lasso_ss_residual / ss_total) # R²
cat("Lasso Valis RMSE:", rmse_valid_lasso, "\n")
cat("Lasso Train RMSE:", rmse_train_lasso, "\n")
cat("Lasso R²:", r_squared_lasso, "%\n")

# Print important variables (non-zero coefficients)
lasso_coef <- coef(lasso_model, s = best_lambda_lasso)
lasso_coef[lasso_coef != 0]

# Ridge (L2) --------------------
set.seed(123)
ridge_model <- glmnet(x_train, y_train, alpha = 0) # alpha = 0 for Ridge

# Cross-validation to find 'optimal' lambda, testing on validation set
#	cv.glmnet performs automatic k-fold cross-validation with default k=10, I change it to 5 for comparison with other models
cv_ridge <- cv.glmnet(x_train, y_train, nfolds = 10, alpha = 0)
best_lambda_ridge <- cv_ridge$lambda.min
ridge_train_pred <- predict(ridge_model, s = best_lambda_ridge, newx = x_train)
ridge_valid_pred <- predict(cv_ridge, s = best_lambda_ridge, newx = x_valid)

# RMSE and R²
rmse_valid_ridge <- sqrt(mean((y_valid - ridge_valid_pred)^2))
rmse_train_ridge <- sqrt(mean((y_train - ridge_train_pred)^2))
ridge_ss_residual <- sum((validation_set$price - ridge_valid_pred)^2)   # RSS
r_squared_ridge <- 1 - (ridge_ss_residual / ss_total) # R²
cat("Ridge Valid RMSE:", rmse_valid_ridge, "\n")
cat("Ridge Train RMSE:", rmse_train_ridge, "\n")
cat("Ridge Train R²:", r_squared_ridge, "%\n")

# Elastic Net (L1 + L2) -----
set.seed(123)
elastic_net_model <- glmnet(x_train, y_train, alpha = 0.5)  # Alpha = 0.5 for Elastic Net

# Cross-validation to find 'optimal' lambda, testing on validation set
cv_elastic <- cv.glmnet(x_train, y_train, nfolds = 10, alpha = 0.5)
best_lambda_elastic <- cv_elastic$lambda.min
elastic_train_pred <- predict(elastic_net_model, s = best_lambda_elastic, newx = x_train)
elastic_valid_pred <- predict(cv_elastic, s = best_lambda_elastic, newx = x_valid)

# RMSE and R²
rmse_valid_elastic <- sqrt(mean((y_valid - elastic_valid_pred)^2))
rmse_train_elastic <- sqrt(mean((y_train - elastic_train_pred)^2))
elastic_ss_residual <- sum((validation_set$price - elastic_valid_pred)^2) # RSS
r_squared_elastic <- 1 - (elastic_ss_residual / ss_total) # R²

cat("Elastic Valis RMSE:", rmse_valid_elastic, "\n")
cat("Elastic Train RMSE:", rmse_train_elastic, "\n")
cat("Elastic R²:", r_squared_elastic, "%\n")

# Bagging -----------------

library(ranger)
set.seed(123)

# Cross-validation
control <- trainControl(
  method = "cv",
  number = 10,
  verboseIter = TRUE,
  allowParallel = TRUE
)

# Tuning grid, ranger requires splitrule and min.node.size therefore:
grid <- expand.grid(
  mtry = floor(ncol(training_set) / 3),  # searching for optimal number of variables at splitting
  splitrule = "variance",   # we minimise variance (but introduce bias)
  min.node.size = c(1, 5)  # Optimal node size tuning
)

#magical mtry value: http://machinelearning202.pbworks.com/w/file/fetch/60606349/breiman_randomforests.pdf

# Bagging training
bagging_model <- caret::train( # overriding caret, sometimes it conflicts with other packages...
  price ~ .,  
  data = training_set,  
  method = "ranger",
  trControl = control,
  tuneGrid = grid,
  ntree = 100,  # 'ranger' does not offer early stopping but it can be achieved through smaller number of trees
  importance = 'impurity',  # ensures feature importance calculation
)

# Print best tuned hyperparameters and features
print(bagging_model$bestTune)
importance <- varImp(bagging_model, scale = FALSE)
# print(importance)  # Identify least important features and remove

importance_plot <- ggplot(importance,
  top = 10,
  measure = "Gain",
  rel_to_first = TRUE
) +
  ggtitle("Bagging Feature Importance (Gain)") + # the importance of each feature relative to the top feature (bedrooms)
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

print(importance_plot)

# Predictions on training and validation set
bagging_train_pred <- predict(bagging_model, newdata = training_set)
bagging_valid_pred <- predict(bagging_model, newdata = validation_set)

# RMSE and R² calculation for both training and validation sets
rmse_train_bagging <- sqrt(mean((training_set$price - bagging_train_pred)^2))
rmse_valid_bagging <- sqrt(mean((validation_set$price - bagging_valid_pred)^2))
bagging_ss_residual <- sum((validation_set$price - bagging_valid_pred)^2) # RSS
r_squared_bagging <- 1 - (bagging_ss_residual / ss_total) # R²
cat("Training RMSE:", rmse_train_bagging, "\n")
cat("Validation RMSE:", rmse_valid_bagging, "\n")
cat("R²:", r_squared_bagging, "\n")

# Combine actual vs predicted values for plotting
plot_data <- data.frame(
  Actual = exp(c(training_set$price, validation_set$price)),
  Predicted = exp(c(bagging_train_pred, bagging_valid_pred)),
  Dataset = rep(c("Training", "Validation"), c(nrow(training_set), nrow(validation_set)))
)

# Plot actual vs predicted values
ggplot(plot_data, aes(x = Actual, y = Predicted, color = Dataset)) +
  geom_point(alpha = 0.6, size = 2) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "black") +
  theme_minimal() +
  labs(
    title = "Bagging Model Performance",
    subtitle = paste("Log Training RMSE:", round(rmse_train_bagging, 2), "| Log Validation RMSE:", round(rmse_valid_bagging, 2), "| R²:", round(r_squared_bagging, 2)),
    x = "Actual Price",
    y = "Predicted Price",
    color = "Dataset"
  ) +
  scale_color_manual(values = c("blue", "red")) +
  theme(legend.position = "top") +
  theme(plot.title = element_text(hjust = 0.5))


# Boosting  ------------------------

# maximum number of trees is set to 200 for all models. CV ensures optimal number of trees
set.seed(123)
boosting_model <- gbm(price ~ ., data = training_set, distribution = "gaussian", 
                      n.trees = 250, interaction.depth = 5, shrinkage = 0.01, 
                      cv.folds = 10, n.cores = NULL, verbose = FALSE)

best_iter <- gbm.perf(boosting_model, method = "cv", plot.it = FALSE)
cat("Optimal number of trees:", best_iter, "\n")

# Generate predictions for training set
boosting_train_pred <- predict(boosting_model, 
                               newdata = training_set, 
                               n.trees = best_iter)

boosting_valid_pred <- predict(boosting_model,
                               newdata = validation_set,
                               n.trees = best_iter)

# I want to compare the perforemence between training and validation models:

# RMSE and R²
rmse_valid_boosting <- sqrt(mean((validation_set$price - boosting_valid_pred)^2))
rmse_train_boosting <- sqrt(mean(training_set$price - boosting_train_pred)^2)
boosting_ss_residual <- sum((validation_set$price - boosting_valid_pred)^2) # RSS
r_squared_boosting <- 1 - (boosting_ss_residual / ss_total) # R²
cat("Boosting training RMSE:", rmse_train_boosting, "\n")
cat("Boosting validation RMSE:", rmse_valid_boosting, "\n")
cat("Boosting training R²:", r_squared_boosting, "%\n")

# Plotting
plot_data <- data.frame(
  Actual = exp(c(training_set$price, validation_set$price)),
  Predicted = exp(c(boosting_train_pred, boosting_valid_pred)),
  Dataset = rep(c("Training", "Validation"), c(nrow(training_set), nrow(validation_set)))
)

ggplot(plot_data, aes(x = Actual, y = Predicted, color = Dataset)) +
  geom_point(alpha = 0.6, size = 2) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "black") +
  theme_minimal() +
  labs(
    title = "Boosting Model Performance",
    subtitle = paste("Log Training RMSE:", round(rmse_train_boosting, 2), "| Log Validation RMSE:", round(rmse_valid_boosting, 2), "| R²:", round(r_squared_boosting, 2)),
    x = "Actual Price",
    y = "Predicted Price",
    color = "Dataset"
  ) +
  scale_color_manual(values = c("blue", "red")) +
  theme(legend.position = "top") +
  theme(plot.title = element_text(hjust = 0.5))

# Random Forest ------------

set.seed(123)
# Cross validation
control <- trainControl(
  method = "cv",           
  number = 10,              # k = 5
  verboseIter = TRUE,      # Print progress during training
  allowParallel = TRUE     # Allow parallel processing if applicable
)

# Define the grid of hyperparameters to tune
grid <- expand.grid(
  mtry = floor((ncol(training_set)-1)/3), # magic paper
  splitrule = "variance",   # we minimise variance (but introduce bias)
  min.node.size = c(1, 5))  # Optimal node size tuning --> equal to bagging for comparison

# Train the Random Forest model using caret with cross-validation
rf_model <- caret::train(
  price ~ .,  
  data = training_set,  
  method = "ranger",           # Random Forest
  trControl = control,     # Cross-validation control
  tuneGrid = grid,         # Grid of hyperparameters to tune
  num.trees = 250,             # no of trees
  importance = "impurity"        # Track feature importance
)

# Print optimal hyperparameters: mtry and ntree
print(rf_model$bestTune)

# Testing for validation set
rf_train_pred <- predict(rf_model, newdata = training_set)
rf_valid_pred <- predict(rf_model, newdata = validation_set)

# RMSE and R² for validation and training sets
rmse_valid_rf <- sqrt(mean((validation_set$price - rf_valid_pred)^2))
rmse_train_rf <- sqrt(mean((training_set$price - rf_train_pred)^2))
rf_ss_residual <- sum((validation_set$price - rf_valid_pred)^2) # RSS
r_squared_rf <- 1 - (rf_ss_residual / ss_total) # R²
cat("Random Forest Validation RMSE:", rmse_valid_rf, "\n")
cat("Random Forest Training RMSE:", r_squared_rf, "\n")

# Plotting
plot_data <- data.frame(
  Actual = exp(c(training_set$price, validation_set$price)),
  Predicted = exp(c(rf_train_pred, rf_valid_pred)),
  Dataset = rep(c("Training", "Validation"), c(nrow(training_set), nrow(validation_set)))
)

ggplot(plot_data, aes(x = Actual, y = Predicted, color = Dataset)) +
  geom_point(alpha = 0.6, size = 2) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "black") +
  theme_minimal() +
  labs(
    title = "Random Forest Model Performance",
    subtitle = paste("Log Training RMSE:", round(rmse_train_rf, 2), "| Log Validation RMSE:", round(rmse_valid_rf, 2), "| R²:", round(r_squared_rf, 2)),
    x = "Actual Price",
    y = "Predicted Price",
    color = "Dataset"
  ) +
  scale_color_manual(values = c("blue", "red")) +
  theme(legend.position = "top") +
  theme(plot.title = element_text(hjust = 0.5))

#Testing residuals:
plot_data$Residuals <- plot_data$Actual - plot_data$Predicted

ggplot(plot_data, aes(x = Actual, y = Residuals, color = Dataset)) +
  geom_point(alpha = 0.6, size = 2) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  theme_minimal() +
  labs(
    title = "Residual Plot: Random Forest Model",
    x = "Actual Price",
    y = "Residuals (Actual - Predicted)",
    color = "Dataset"
  ) +
  scale_color_manual(values = c("blue", "red")) +
  theme(legend.position = "top")

# Error distribution for Random Forest (Validation Set only):
plot_data_valid <- subset(plot_data, Dataset == "Validation") # Filter data for validation set only
histogram_plot <- ggplot(plot_data_valid, aes(x = Residuals, fill = Dataset)) +
  geom_histogram(alpha = 0.5, binwidth = 10, position = "identity") +
  theme_minimal() +
  labs(
    title = "Error Distribution: Random Forest Model (Validation Set)",
    x = "Prediction Error (Residuals)",
    y = "Count",
    fill = "Dataset"
  ) +
  scale_fill_manual(values = c("red")) +
  theme(legend.position = "top")
print(histogram_plot) # remember to print

# Checking for normality of errors:
library(ggpubr)
ggqqplot(plot_data$Residuals) +
  theme_minimal() +
  labs(title = "QQ Plot of Residuals for Random Forest", subtitle = "Checking Normality of Errors Distribution")

# Density scatter plot
ggplot(plot_data, aes(x = Actual, y = Predicted)) +
  geom_hex(bins = 35) +  # Hexbin density plot
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "white") +
  scale_fill_viridis_c() +  # Nice gradient color
  theme_minimal() +
  labs(
    title = "Model Performance: Actual vs. Predicted (Density View)",
    x = "Actual Price",
    y = "Predicted Price"
  )

# XGBoost: -----------

set.seed(123)
# Convert datasets to DMatrix format for XGBoost
train_matrix <- as.matrix(training_set[, -which(names(training_set) == "price")])
train_label <- training_set$price
valid_matrix <- as.matrix(validation_set[, -which(names(validation_set) == "price")])
valid_label <- validation_set$price

dtrain <- xgb.DMatrix(data = train_matrix, label = train_label)
dvalid <- xgb.DMatrix(data = valid_matrix, label = valid_label)

# XGBoost parameters
params <- list(
  objective = "reg:squarederror", # regression problem therefore minimize the squared error
  eta = 0.05, # learning rate = 0.05
  max_depth = 10, # limiting complexity
  subsample = 0.7, # avoiding overfitting
  colsample_bytree = 0.7,
  eval_metric = "rmse"
)

# Cross-validation with early stopping
cv_results <- xgb.cv(
  params = params,
  data = dtrain,
  nfold = 10,
  nrounds = 250, # equal ntrees for all models
  early_stopping_rounds = 50, # avoiding overfitting and computational complexity
  verbose = 5 # 0 - no printing of progress
)

best_nrounds <- cv_results$best_iteration # choosing optimal hyperparameters
cat("Best number of rounds:", best_nrounds, "\n")

# Training final model
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = best_nrounds,
  watchlist = list(train = dtrain, valid = dvalid),
  early_stopping_rounds = 50,
  verbose = 1
)

# Predictions and evaluation
xgb_train_pred <- predict(xgb_model, dtrain)
xgb_valid_pred <- predict(xgb_model, dvalid)

rmse_train_xgb <- sqrt(mean((train_label - xgb_train_pred)^2))
rmse_valid_xgb <- sqrt(mean((valid_label - xgb_valid_pred)^2))
xgb_ss_residual <- sum((validation_set$price - xgb_valid_pred)^2) # RSS
r_squared_xgb <- 1 - (xgb_ss_residual / ss_total) # R²
cat("XGBoost Training RMSE:", rmse_train_xgb, "\n")
cat("XGBoost Validation RMSE:", rmse_valid_xgb, "\n")
cat("XGBoost Training R²:", r_squared_xgb, "%\n")

# Loading plotting data and plotting
plot_data <- data.frame(
  Actual = exp(c(training_set$price, validation_set$price)),
  Predicted = exp(c(xgb_train_pred, xgb_valid_pred)),
  Dataset = rep(c("Training", "Validation"), c(nrow(training_set), nrow(validation_set)))
)
plot_data$Residuals <- plot_data$Actual - plot_data$Predicted


# Checking for normality of errors:
library(ggpubr)
ggqqplot(plot_data$Residuals) +
  theme_minimal() +
  labs(title = "QQ Plot of Residuals for XGBoost", subtitle = "Checking Normality of Errors Distribution")

# General Scatter plot
ggplot(plot_data, aes(x = Actual, y = Predicted, color = Dataset)) +
  geom_point(alpha = 0.6, size = 2) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "black") +
  theme_minimal() +
  labs(
    title = "XGBoost Model Performance",
    subtitle = paste("Log Training RMSE:", round(rmse_train_xgb, 2), "| Log Validation RMSE:", round(rmse_valid_xgb, 2), "| R-squared", round(r_squared_xgb, 3)),
    x = "Actual Price",
    y = "Predicted Price",
    color = "Dataset"
  ) +
  scale_color_manual(values = c("blue", "red")) +
  theme(legend.position = "top") +
  theme(plot.title = element_text(hjust = 0.5))

# Bayesian XGBoost ------------
# The following code is based on rBayesianOptimization package example
new_xgb_cv_bayes <- function(max_depth, eta, subsample, colsample_bytree, min_child_weight) 
  {
  
  max_depth <- round(max_depth)  # Ensure max_depth is an integer
  
  params <- list(
    objective = "reg:squarederror",
    eval_metric = "rmse", #the hyperparameters are aimed to minimise RMSE (penalise big mistakes)
    max_depth = max_depth,
    eta = eta,
    subsample = subsample,
    colsample_bytree = colsample_bytree,
    min_child_weight = min_child_weight
  )
  
  # k-Fold Cross-Validation
  cv_results <- xgb.cv(
    params = params,
    data = dtrain,
    nrounds = 250,
    nfold = 10,
    early_stopping_rounds = 50, #ensures computational efficiency and limits overfitting
    verbose = 0
  )
  
  # Negative RMSE since rBayesianOptimization maximizes the objective
  return(list(Score = -min(cv_results$evaluation_log$test_rmse_mean), Pred = 0))
}

# rBayesianOptimization
set.seed(123)
opt_results <- BayesianOptimization(
  FUN = new_xgb_cv_bayes,
  bounds = list(
    max_depth = c(5L, 15L),
    eta = c(0.01, 0.3),
    subsample = c(0.5, 1),
    colsample_bytree = c(0.5, 1),
    min_child_weight = c(1, 10)
  ),
  init_points = 10,
  n_iter = 20,
  acq = "ei",
  verbose = TRUE
)

# Listing the best hyperparameters
new_xgb_best_params <- list(
  max_depth = round(opt_results$Best_Par["max_depth"]),
  eta = opt_results$Best_Par["eta"],
  subsample = opt_results$Best_Par["subsample"],
  colsample_bytree = opt_results$Best_Par["colsample_bytree"],
  min_child_weight = opt_results$Best_Par["min_child_weight"],
  objective = "reg:squarederror",
  eval_metric = "rmse"
)
cat("Best Hyperparameters: \n")
print(new_xgb_best_params)

# Final model on the training set
new_xgb_model <- xgb.train(
  params = new_xgb_best_params,
  data = dtrain,
  nrounds = 250,
  watchlist = list(train = dtrain, valid = dvalid),  # Now includes validation set
  early_stopping_rounds = 50,
  verbose = 1
)

# RMSE and R²
new_xgb_train_pred <- predict(new_xgb_model, dtrain)
new_xgb_valid_pred <- predict(new_xgb_model, dvalid)

rmse_train_new_xgb <- sqrt(mean((train_label - new_xgb_train_pred)^2))
rmse_valid_new_xgb <- sqrt(mean((valid_label - new_xgb_valid_pred)^2))
new_xgb_ss_residual <- sum((validation_set$price - new_xgb_valid_pred)^2) # RSS
r_squared_new_xgb <- 1 - (new_xgb_ss_residual / ss_total) # R²

cat("Bayesian XGBoost Training RMSE:", rmse_train_new_xgb, "\n")
cat("Bayesian XGBoost Validation RMSE:", rmse_valid_new_xgb, "\n")
cat("Bayesian XGBoost R²:", r_squared_new_xgb, "%\n")

# Bayesian XGBoost graphics -----------

# Plotting
plot_data <- data.frame(
  Actual = exp(c(training_set$price, validation_set$price)),
  Predicted = exp(c(new_xgb_train_pred, new_xgb_valid_pred)),
  Dataset = rep(c("Training", "Validation"), c(nrow(training_set), nrow(validation_set)))
)

ggplot(plot_data, aes(x = Actual, y = Predicted, color = Dataset)) +
  geom_point(alpha = 0.6, size = 2) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "black") +
  theme_minimal() +
  labs(
    title = "Bayesian Optimised XGBoost Model Performance",
    subtitle = paste("Log Training RMSE:", round(rmse_train_new_xgb, 2), "| Log Validation RMSE:", round(rmse_valid_new_xgb, 2), "| R²:", round(r_squared_new_xgb, 3)),
    x = "Actual Price",
    y = "Predicted Price",
    color = "Dataset"
  ) +
  scale_color_manual(values = c("blue", "red")) +
  theme(legend.position = "top") +
  theme(plot.title = element_text(hjust = 0.5))

# Frankenstein Model ----------

# Load necessary package
library(nloptr)  # Optimization package to ensure all weights sum to one.

# Defining loss function as RMSE
frankenstein_loss_function <- function(p, price, xgb_train_pred, rf_train_pred, lasso_train_pred) {
  price_pred <- p[1] * xgb_train_pred + p[2] * rf_train_pred + p[3] * lasso_train_pred
  sqrt(mean((price - price_pred)^2))
}

# Initial weights distribution
initial_weights <- c(1/3, 1/3, 1/3)
#ensure equal dimensions:
# length(training_set$price)
# length(xgb_train_pred)
# length(rf_train_pred)
# length(lasso_train_pred)

# 'optimal' weights
opt_result <- nloptr(
  x0 = initial_weights,
  eval_f = function(p) frankenstein_loss_function(p, training_set$price, xgb_train_pred, rf_train_pred, lasso_train_pred),
  lb = c(0, 0, 0),  # Non-negative weights
  ub = c(1, 1, 1),  # Optional upper bounds
  eval_g_eq = function(p) c(sum(p) - 1),  # Ensure sum of weights = 1
  opts = list(algorithm = "NLOPT_LN_COBYLA", xtol_rel = 1.0e-6)
)

# Training RMSE for Lasso is ridiculous and bad but decent for Validation - improve the weights

# Print optimal weights
optimal_weights <- opt_result$solution
print(optimal_weights)

# Final predictions
frankenstein_valid_pred <- optimal_weights[1] * xgb_valid_pred +
  optimal_weights[2] * rf_valid_pred +
  optimal_weights[3] * lasso_valid_pred

# RMSE and R-squared
rmse_valid_frankenstein <- sqrt(mean((validation_set$price - frankenstein_valid_pred)^2))
frankenstein_ss_residual <- sum((validation_set$price - frankenstein_valid_pred)^2) # RSS
r_squared_frankenstein <- 1 - (frankenstein_ss_residual / ss_total) # R²
cat("Random Frankenstein RMSE:", rmse_valid_frankenstein, "\n")
cat("Random Frankenstein R²:", r_squared_frankenstein, "%\n")

# Testing with non optimised weights (does not yield any results)
final_pred <- 0.7 * xgb_valid_pred + 0.3 * lasso_valid_pred
rmse_valid_second <- sqrt(mean((validation_set$price - final_pred)^2))
cat("testing:", rmse_valid_second, "\n")




# SUMMARY ------------------------
cat("Model RMSE Scores:\n")
cat("Ridge Regression:", rmse_ridge, r_squared_ridge, "\n")
cat("Lasso Regression:", rmse_valid_lasso, r_squared_lasso, "\n")
cat("Elastic Net:", rmse_elastic, r_squared_elastic, "\n")
cat("Bagging:", rmse_valid_bagging, r_squared_bagging, "\n")
cat("Random Forest:", rmse_valid_rf, r_squared_rf, "\n")
cat("Boosting (GBM):", rmse_valid_boosting, r_squared_boosting, "\n")
cat("XGBoost:", rmse_valid_xgb, r_squared_xgb, "\n")
cat("Bayesian XGBoost:", rmse_valid_new_xgb, r_squared_new_xgb, "\n")
cat("'Frankenstein Model':", rmse_valid_frankenstein, r_squared_frankenstein, "\n")

# Overfitting and performance tables ----- 

# "rmse_valid" index comparison
rmse_valid_vars = ls(pattern = "^rmse_valid")  # Finding all variables starting with "rmse_valid"
rmse_valid_df = data.frame(
  method = sub("^rmse_valid_", "", rmse_valid_vars),  # Removing the prefix
  rmse = sapply(rmse_valid_vars, function(x) get(x)),
  stringsAsFactors = FALSE
)
rmse_valid_df = rmse_valid_df %>%
  arrange(rmse)

# "rmse_train" index comparison
rmse_train_vars = ls(pattern = "^rmse_train")  # Finding all variables starting with "rmse_train"
rmse_train_df = data.frame(
  method = sub("^rmse_train_", "", rmse_train_vars),  # Removing the prefix
  rmse = sapply(rmse_train_vars, function(x) get(x)),
  stringsAsFactors = FALSE
)
rmse_train_df = rmse_train_df %>%
  arrange(rmse)

# "R-squared" index comparison
r_squared_vars = ls(pattern = "^r_squared")  # Finding all variables starting with "r_squared_valid"
r_squared_df = data.frame(
  method = sub("^r_squared_", "", r_squared_vars),  # Removing the prefix
  r_squared = sapply(r_squared_vars, function(x) get(x)),
  stringsAsFactors = FALSE
)
r_squared_df = r_squared_df %>%
  arrange(r_squared)

# View the resulting data frames
View(rmse_train_df)
View(rmse_valid_df)
View(r_squared_df)

# Final table  -----
# Merging all final dataframes
performance_table <- merge(
  merge(
    rmse_train_df,
    rmse_valid_df,
    by = "method",  # Merge train and valid RMSE first
    suffixes = c("_train", "_valid")
  ),
  r_squared_df,
  by = "method"  # Then merge with R-squared
)
colnames(performance_table) <- c("Model", "RMSE (Training)", "RMSE (Validation)", "R-squared")

# Generating LaTeX table (hopefully)
final_performance_table <- knitr::kable(
  performance_table,
  format = "latex",
  caption = "Model Performance Comparison",
  booktabs = TRUE,
  align = c("l", "r", "r", "r"),  
  digits = c(0, 3, 3, 3),         # 3 decimal places are enough
  col.names = c("Model", "RMSE (Training)", "RMSE (Validation)", "R-squared")
)
cat(final_performance_table)

# To reload the tables please remove the dataframes and rerun code:
# sapply(rmse_valid_vars, function(x) length(get(x))) #verify there are not conflicting arrays
if(exists("rmse_valid_df")) rm(rmse_valid_df) # remove them if necessary
if(exists("rmse_valid_vars")) rm(rmse_valid_vars) 
if(exists("rmse_train_df")) rm(rmse_train_df) 
if(exists("rmse_train_vars")) rm(rmse_train_vars)
if(exists("r_squared_df")) rm(r_squared_df) 
if(exists("r_squared_vars")) rm(r_squared_vars) 

# GRAPHS AND TABLES -----
# XGBoost Feature importance graphics -----------

# XGBoost package allows for feature importance analysis
# Plotting importance features
library(Ckmeans.1d.dp)
importance_matrix <- xgb.importance(
  feature_names = colnames(train_matrix),
  model = xgb_model
)

importance_plot <- xgb.ggplot.importance(
  importance_matrix = importance_matrix,
  top_n = 10,
  measure = "Gain", # Additionally: "Cover" is the relative number of observations related to this feature or "Frequency" is the number of times a feature is used in all generated trees
  rel_to_first = TRUE
) +
  ggtitle("XGBoost Feature Importance (Gain)") + # the importance of each feature relative to the top feature (bedrooms)
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

print(importance_plot)

# Bin -------------

# Remove all the local dataframeworks for data storage efficiency:

# FNN Testing ---------------------
# Load required libraries
library(caret)
library(ggplot2)

# Set seed for reproducibility
set.seed(123)

# Cross-validation setup (same as RF)
control <- trainControl(
  method = "cv",           
  number = 10,              # 10-fold CV
  verboseIter = TRUE,       # Print training progress
  allowParallel = TRUE      # Enable parallel processing
)

# Hyperparameter grid for FNN
grid <- expand.grid(
  layer1 = c(16, 32, 64),     # Neurons in first hidden layer
  layer2 = c(8, 16),          # Neurons in second hidden layer (optional)
  layer3 = c(0),              # No third layer (set to 0 to disable)
  activation = "relu",        # Activation function (ReLU for regression)
  dropout = c(0.1, 0.2),      # Dropout rate to prevent overfitting
  learning_rate = c(0.001, 0.01),  # Learning rate
  batch_size = c(32, 64),     # Mini-batch size
  epochs = c(50, 100)         # Training epochs
)

# Train the FNN model using caret
fnn_model <- caret::train(
  price ~ .,  
  data = training_set,  
  method = "mlpKerasDropout",           # Multi-layer perceptron (FNN)
  trControl = control,        # Cross-validation
  tuneGrid = grid,            # Hyperparameter grid
  preProcess = c("center", "scale"),  # Standardize data
  metric = "RMSE",            # Optimize for RMSE
)

# Print optimal hyperparameters
print(fnn_model$bestTune)
fnn_train_pred <- predict(fnn_model, newdata = training_set)
fnn_valid_pred <- predict(fnn_model, newdata = validation_set)

# RMSE and R²
rmse_valid_fnn <- sqrt(mean((validation_set$price - fnn_valid_pred)^2))
rmse_train_fnn <- sqrt(mean((training_set$price - fnn_train_pred)^2))
cat("FNN Validation RMSE:", rmse_valid_fnn, "\n")
cat("FNN Training RMSE:", rmse_train_fnn, "\n")

# Plotting
plot_data <- data.frame(
  Actual = c(training_set$price, validation_set$price),
  Predicted = c(fnn_train_pred, fnn_valid_pred),
  Dataset = rep(c("Training", "Validation"), c(nrow(training_set), nrow(validation_set)))
)

ggplot(plot_data, aes(x = Actual, y = Predicted, color = Dataset)) +
  geom_point(alpha = 0.6, size = 2) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "black") +
  theme_minimal() +
  labs(
    title = "Feedforward Neural Network Performance",
    subtitle = paste("Training RMSE:", round(rmse_train_fnn, 2), 
                     "| Validation RMSE:", round(rmse_valid_fnn, 2),
    x = "Actual Price",
    y = "Predicted Price",
    color = "Dataset"
  ) +
  scale_color_manual(values = c("blue", "red")) +
  theme(legend.position = "top") +
  theme(plot.title = element_text(hjust = 0.5)))

