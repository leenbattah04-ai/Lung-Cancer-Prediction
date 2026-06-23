# Lung Cancer Prediction and Classification

# Load Libraries
library(corrplot)
library(car)
library(cluster)
library(caret)
library(e1071)
library(rpart)
library(rpart.plot)
library(class)

# Create Plots Folder
dir.create("plots", showWarnings = FALSE)

# Read Dataset
cancer_data <- read.csv("Lung Cancer Dataset.csv")

head(cancer_data)
str(cancer_data)
dim(cancer_data)

# Select Numerical Features
numeric_cols <- cancer_data[, sapply(cancer_data, is.numeric)]

ncol(numeric_cols)
names(numeric_cols)

# Missing Values
any(is.na(cancer_data))
colSums(is.na(cancer_data))


# Target Variable
cancer_data$PULMONARY_DISEASE <- as.factor(cancer_data$PULMONARY_DISEASE)

summary(cancer_data$PULMONARY_DISEASE)

round(prop.table(table(cancer_data$PULMONARY_DISEASE)) * 100, 2)

# Descriptive Statistics

Stats <- function(col) {
  c(
    Min = min(col, na.rm = TRUE),
    Max = max(col, na.rm = TRUE),
    Mean = mean(col, na.rm = TRUE),
    StdDev = sd(col, na.rm = TRUE),
    Q1 = quantile(col, 0.25, na.rm = TRUE),
    Q3 = quantile(col, 0.75, na.rm = TRUE)
  )
}

stats_result <- as.data.frame(t(sapply(numeric_cols, Stats)))
stats_result

# Percentile Plots
png("plots/percentile_plots.png", width = 1000, height = 1000)

par(mfrow = c(3, 6))

for (i in names(numeric_cols)) {
  
  plot(
    ecdf(numeric_cols[[i]]),
    main = paste("Percentile Plot of", i),
    xlab = i,
    ylab = "Cumulative Probability"
  )
  
}

dev.off()

# Histograms
png("plots/histograms_lung.png", width = 1000, height = 1000)

par(mfrow = c(3, 6))

for (i in names(numeric_cols)) {
  
  hist(
    numeric_cols[[i]],
    main = paste("Histogram of", i),
    xlab = i,
    col = "purple",
    border = "black"
  )
  
}

dev.off()

# Boxplots
png("plots/boxplots_lung.png", width = 1000, height = 1000)

par(mfrow = c(3, 6))

for (i in names(numeric_cols)) {
  
  boxplot(
    numeric_cols[[i]],
    main = paste("Boxplot of", i),
    ylab = i,
    col = "yellow"
  )
  
}

dev.off()

# Pie Chart
png("plots/pie_pulmonary_disease.png", width = 600, height = 600)

pie(
  table(cancer_data$PULMONARY_DISEASE),
  main = "Pulmonary Disease Distribution",
  col = c("lightblue", "pink")
)

dev.off()

# Correlation Matrix
cor_matrix <- cor(numeric_cols, use = "complete.obs")

png("plots/correlation_matrix_lung.png",
    width = 1000,
    height = 900)

corrplot(
  cor_matrix,
  method = "color",
  type = "upper",
  tl.col = "black",
  tl.cex = 0.8,
  number.cex = 0.7
)

dev.off()

# Scatterplot Matrix
png("plots/scatterplot_matrix_lung.png",
    width = 1000,
    height = 1000)

scatterplotMatrix(
  numeric_cols,
  main = "Matrix Scatterplot for Lung Cancer Dataset"
)

dev.off()

# K-Means Clustering
scaled_data <- scale(numeric_cols)

set.seed(123)

for (k in 2:4) {
  
  k_model <- kmeans(scaled_data, centers = k)
  
  png(
    paste0("plots/clusplot_k", k, ".png"),
    width = 800,
    height = 600
  )
  
  clusplot(
    scaled_data,
    k_model$cluster,
    color = TRUE,
    shade = TRUE,
    labels = 2,
    lines = 0,
    main = paste("K-Means Clustering with K =", k)
  )
  
  dev.off()
  
}

# Supervised Machine Learning
y <- cancer_data$PULMONARY_DISEASE
x <- numeric_cols

# Train-Test Split
set.seed(123)

train_index <- createDataPartition(
  y,
  p = 0.7,
  list = FALSE
)

x_train <- x[train_index, ]
x_test <- x[-train_index, ]

y_train <- y[train_index]
y_test <- y[-train_index]

# Naive Bayes
nb_model <- naiveBayes(x_train, y_train)
nb_pred <- predict(nb_model, x_test)

# Decision Tree
dt_model <- rpart(
  y_train ~ .,
  data = data.frame(x_train, y_train),
  method = "class"
)

dt_pred <- predict(
  dt_model,
  x_test,
  type = "class"
)

png("plots/decision_tree_lung.png",
    width = 800,
    height = 600)

rpart.plot(dt_model)

dev.off()

# KNN
knn_pred <- knn(
  train = scale(x_train),
  test = scale(x_test),
  cl = y_train,
  k = 5
)

# SVM
svm_model <- svm(
  scale(x_train),
  y_train,
  kernel = "radial"
)

svm_pred <- predict(
  svm_model,
  scale(x_test)
)

# Evaluation Function
evaluate_model <- function(pred, true) {
  
  cm <- confusionMatrix(pred, true)
  
  accuracy <- cm$overall["Accuracy"]
  
  data.frame(
    Accuracy = accuracy,
    Classification_Error = 1 - accuracy,
    Precision = cm$byClass["Pos Pred Value"],
    Recall = cm$byClass["Sensitivity"],
    F1 = cm$byClass["F1"]
  )
  
}

# Compare Models

results <- rbind(
  
  NaiveBayes = evaluate_model(nb_pred, y_test),
  
  DecisionTree = evaluate_model(dt_pred, y_test),
  
  KNN = evaluate_model(knn_pred, y_test),
  
  SVM = evaluate_model(svm_pred, y_test)
  
)

print(results)

# Save Results

write.csv(
  results,
  "LungCancer_ML_Results.csv",
  row.names = TRUE
)