# Load required libraries
library(caret)
library(randomForest)
library(e1071)
library(xgboost)
library(tidyverse)
library(knitr)
library(readr)

# Load the dataset
user_data <- read_csv("R/CustBehavior.csv", 
                      col_types = cols(
                        `User ID` = col_double(),
                        Age = col_double(),
                        Gender = col_character(),
                        Country = col_character(),
                        `Purchase Amount` = col_double(),
                        `Purchase Date` = col_date(),
                        `Product Category` = col_character()
                      ))

# Rename columns to avoid spaces
colnames(user_data) <- make.names(colnames(user_data))

# Check and handle missing values
sum(is.na(user_data))
user_data <- user_data[complete.cases(user_data), ]

# Prepare the data
user_data$Purchase.Date <- as.Date(user_data$Purchase.Date)
user_data$Gender <- as.factor(user_data$Gender)
user_data$Country <- as.factor(user_data$Country)
user_data$Product.Category <- as.factor(user_data$Product.Category)

# Define target and feature columns
target_col <- "Purchase.Amount"
feature_cols <- c("User.ID", "Age", "Gender", "Country", "Product.Category")

# Split data into training and testing sets
set.seed(123)
train_index <- createDataPartition(user_data[[target_col]], p = 0.8, list = FALSE)
train_data <- user_data[train_index, ]
test_data <- user_data[-train_index, ]

# Helper function to evaluate model performance
evaluate_model <- function(predictions, actual, model_name) {
  rmse <- RMSE(predictions, actual)
  mae <- MAE(predictions, actual)
  data.frame(Model = model_name, RMSE = rmse, MAE = mae)
}

# 1. Linear Regression
lm_model <- train(
  as.formula(paste(target_col, "~", paste(feature_cols, collapse = " + "))),
  data = train_data,
  method = "lm",
  trControl = trainControl(method = "cv", number = 5)
)
lm_pred <- predict(lm_model, test_data)
lm_metrics <- evaluate_model(lm_pred, test_data[[target_col]], "Linear Regression")

# 2. Random Forest
rf_model <- train(
  as.formula(paste(target_col, "~", paste(feature_cols, collapse = " + "))),
  data = train_data,
  method = "rf",
  trControl = trainControl(method = "cv", number = 5)
)
rf_pred <- predict(rf_model, test_data)
rf_metrics <- evaluate_model(rf_pred, test_data[[target_col]], "Random Forest")

# 3. Support Vector Machine (SVM)
svm_model <- train(
  as.formula(paste(target_col, "~", paste(feature_cols, collapse = " + "))),
  data = train_data,
  method = "svmRadial",
  trControl = trainControl(method = "cv", number = 5)
)
svm_pred <- predict(svm_model, test_data)
svm_metrics <- evaluate_model(svm_pred, test_data[[target_col]], "SVM")

# 4. XGBoost
xgb_model <- train(
  as.formula(paste(target_col, "~", paste(feature_cols, collapse = " + "))),
  data = train_data,
  method = "xgbTree",
  trControl = trainControl(method = "cv", number = 5)
)
xgb_pred <- predict(xgb_model, test_data)
xgb_metrics <- evaluate_model(xgb_pred, test_data[[target_col]], "XGBoost")

# Combine model performance metrics into a data frame
performance_df <- rbind(lm_metrics, rf_metrics, svm_metrics, xgb_metrics)

# Print the performance results
print(performance_df)

# Save performance results to CSV
write.csv(performance_df, "model_performance_results.csv", row.names = FALSE)

# Generate and save the performance plot
performance_plot <- ggplot(performance_df, aes(x = Model, y = RMSE)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  theme_minimal() +
  coord_flip() +
  labs(title = "Model Performance Comparison", y = "RMSE")

# Display the plot
print(performance_plot)

# Save the plot as an image
ggsave("model_performance_plot.png", plot = performance_plot, width = 8, height = 5)

# Save the models and performance metrics for report generation
save(lm_model, rf_model, svm_model, xgb_model, performance_df, performance_plot, file = "user_transaction_analysis.RData")

