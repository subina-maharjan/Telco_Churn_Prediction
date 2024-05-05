library(dplyr)  # For data manipulation
library(readr)  # For reading data
library(tidyr) 
library(caret) # For confusion matrix


telco <- read_csv("Telco_customer_churn.csv")
library(janitor)
library(dplyr)

telco <- telco%>% clean_names()
names(telco)

# Read status.csv file
status <- read_csv("Telco_customer_churn_status.csv")
status <- status%>% clean_names()
names(status)

services <- read_csv("Telco_customer_churn_services.csv")
services <- services%>% clean_names()
names(services)
head(services)

names(telco)
head(telco)


# Merge satisfaction_score column from status dataframe into telco dataframe
telco <- merge(telco, status[, c("customer_id", "satisfaction_score")], by = "customer_id", all.x = TRUE)
telco <- merge(telco, services[, c("customer_id",  "total_refunds", "total_extra_data_charges", 
                                   "total_long_distance_charges", "referred_a_friend","number_of_referrals")], by = "customer_id", all.x = TRUE)

dim(telco)
names(telco)



str(telco)

# Remove the 'churn_reason' column from the dataframe
telco <- telco %>% 
  select(-churn_reason,-churn_score,-churn_label,-count, -customer_id,-city,-state,-country,-lat_long)

dim(telco)
# Convert categorical variables to factors
telco <- telco %>%
  mutate_if(is.character, as.factor)

# Convert churn_label to factor
#telco$churn_label <- as.factor(telco$churn_label)

dim(telco)
# Handle missing values (if necessary)
telco <- na.omit(telco)  # This removes rows with any NA values
dim(telco)

str(telco)
dim(telco)

attach(telco)
churn <- ifelse(churn_value == 1, "Yes", "NO")
churn <- as.factor(churn)



telco <- data.frame(telco,churn)
dim(telco)

 #BOX PLOT with satisfaction score and churn level
library(ggplot2)

# Create a box plot 
ggplot(telco, aes(x = churn, y = satisfaction_score, fill = churn)) +
  geom_boxplot() +  # Use geom_violin() for violin plot
  labs(x = "Churn", y = "Satisfaction Score", fill = "Churn Status") +
  ggtitle("Satisfaction Score by Churn Status")


## Bar plot with contract type and churn level


# Create a sample dataset (replace this with your actual dataset)
data <- data.frame(contract,churn)


# Count frequencies of churn by contract
frequency_table <- table(data$contract, data$churn)
# Calculate total customers for each contract type
total_customers <- sum(frequency_table)

# Calculate proportions
proportion_table <- frequency_table / total_customers

# Create a grouped bar plot with percentages
barplot(proportion_table, beside = TRUE, legend.text = TRUE,
        main = "Churn Frequency by Contract",
        xlab = "Contract",
        ylab = "Frequency (%)",
        col = c("lightblue", "salmon", "lightgreen"))


##Decision Tree

n <- nrow(telco)
n
set.seed(2)

def.subset <- sample(n, size = (0.8*n))

# Create train and test data
train.data <- telco[def.subset, ]
test.data <- telco[-def.subset,]
churn.test <- churn[-def.subset]

dim(train.data)
dim(test.data)
dim(churn.test)

library(tree)
str(telco)

dim(telco)
head(telco)

tree_formula <- churn ~  senior_citizen + contract + dependents + tenure_months + phone_service + 
  multiple_lines + internet_service + online_security + online_backup + 
  device_protection + tech_support + streaming_tv + streaming_movies  + payment_method + total_charges + total_refunds + 
  total_extra_data_charges + total_long_distance_charges + number_of_referrals +
  referred_a_friend +zip_code


tree.telco <- tree(tree_formula, data = train.data)


# We graphically display the fitted classification tree:
par(mfrow=c(1,1))
plot(tree.telco)
text(tree.telco, pretty = 0)

summary(tree.telco)

# The tree was constructed using  "contract" "dependents"    "number of referrals"  "tenure_months" "internet_service" "total_long_distance_charges"
# the misclassifcation error rate is 0.20 , which means that approximately 20%
# of the instances in the dataset were misclassified by the model.


#Test error

tree.pred = predict(tree.telco, test.data, type = "class")

tab <- table(tree.pred, churn.test)
tab

# test error
(tab[1,2] + tab[2,1]) / sum(tab)  
#0.21

# Compute confusion matrix
conf_matrix0 <- confusionMatrix(tree.pred, churn.test)

# Extract accuracy
accuracy0 <- conf_matrix0$overall["Accuracy"]
accuracy0
#78.82

#. Leave-one-out cross-validation

# An alternative way to asses the test error.

n <- nrow(telco)  
cv.predictions <- rep('Yes', n)

for(i in 1:n) { # start a loop over all data points
  # Fit a classification tree using all data except one data point:
  tree.fit <- tree(tree_formula, data = telco[-i, ]) 
  
  # Make a prediction for the excluded data point:
  cv.predictions[i] <- predict(tree.fit, newdata = telco[i,], type = "class")
} 

# Now, we compare the predictions with the actual class labels:
tab <- table(cv.predictions, churn)
tab


cv.error = (tab[1,2] + tab[2,1]) / sum(tab) 
cv.error
#0.20

# Pruning the tree

set.seed(3)
cv.telco <- cv.tree(tree.telco, FUN = prune.misclass)
names(cv.telco)

plot(cv.telco$size, cv.telco$dev, type = "b", 
     xlab = "The number of terminal nodes", 
     ylab = "The number of misclassified points")

best.size <- cv.telco$size[which.min(cv.telco$dev)]
best.size
#9

prune.telco <- prune.misclass(tree.telco, best = best.size)
plot(prune.telco)
text(prune.telco, pretty = 0)

# How well does this pruned tree perform on the test data? 

tree.pred <- predict(prune.telco, test.data, type = "class") 
tab <- table(tree.pred, churn.test) 
(tab[1,2] + tab[2,1]) / sum(tab)
#test error
#0.21

conf_matrix.table <- tab
precision.tree <- diag(conf_matrix.table) / rowSums(conf_matrix.table)
recall.tree <- diag(conf_matrix.table) / colSums(conf_matrix.table)
F1_score.tree <- 2 * (precision.tree * recall.tree) / (precision.tree + recall.tree)


# random forest
library(randomForest)

set.seed(4)
rf.tree <- randomForest(tree_formula, data = train.data, 
                         mtry = 6, importance = TRUE) 

# The argument mtry=6 indicates that all 6 predictors should 
# be considered for each split of the tree - in other words, 
# that bagging should be done. total features is 36 and square root of 36 is 6.

# How well does this bagged model perform on the test set?

rf.pred <- predict(rf.tree, test.data, type = "class") 

tab <- table(rf.pred, churn.test) 
tab

(tab[1,2] + tab[2,1]) / sum(tab)
#0.19

# Compute confusion matrix
conf_matrix <- confusionMatrix(rf.pred, churn.test)

# Extract accuracy
accuracy <- conf_matrix$overall["Accuracy"]
accuracy
#80.67

# Compute variable importance
varImpPlot(rf.tree, n.var= 10,type=1,main = "Feature Importance")

summary(rf.tree)
# Random forest

plot(rf.tree)
## Calculate precision, recall, and F1 score
precision.rf50 <- diag(tab) / rowSums(tab)
recall.rf50 <- diag(tab) / colSums(tab)
F1_score.rf50 <- 2 * (precision.rf50 * recall.rf50) / (precision.rf50 + recall.rf50)


str(telco)
#--------------------Optimize probability threshold cutoff-----------
# Make predictions based on Probability
rf.Probability <- predict(rf.tree, test.data, type = "prob")
# Extract probabilities of the positive class(Churned)
prob_positive_class <- rf.Probability[, "Yes"]

# Try different threshold cutoffs and calculate 
# missed churns ratio, all predicted churns ratio, FalsePositive, 
# and TruePositive for each threshold
Prob_Threshold <- seq(0.1, 0.9, by = 0.01) #Probability threshold
missed_churns <- numeric(length = length(Prob_Threshold))
predicted_churns <- numeric(length = length(Prob_Threshold))
FalsePositive <- numeric(length = length(Prob_Threshold))
TruePositive <- numeric(length = length(Prob_Threshold))

precision.Prob <- numeric(length = length(Prob_Threshold))
recall.Prob <- numeric(length = length(Prob_Threshold))
F1_score.Prob <- numeric(length = length(Prob_Threshold))
index.predict <- 1
for (i in Prob_Threshold) {
  positive_class <- ifelse(prob_positive_class > i , "Yes", "NO") #threshold cutoffs
  confusion_matrix <- table(positive_class, test.data$churn)
  missed_churns[index.predict] <- (confusion_matrix[1,2]/sum(confusion_matrix))
  predicted_churns[index.predict] <- ((confusion_matrix[2,1] + confusion_matrix[2,2])/sum(confusion_matrix))
  FalsePositive[index.predict] <- confusion_matrix[2,1]/(confusion_matrix[2,1]+confusion_matrix[1,1])
  TruePositive[index.predict] <- confusion_matrix[2,2]/(confusion_matrix[2,2]+confusion_matrix[1,2])
  # F1-score
  TP <- confusion_matrix[2, 2]
  FP <- confusion_matrix[1, 2]
  FN <- confusion_matrix[2, 1]
  precision <- TP / (TP + FP)
  recall <- TP / (TP + FN)
  F1_score.Prob[index.predict] <- 2 * (precision * recall) / (precision + recall)
  precision.Prob[index.predict] <- precision
  recall.Prob[index.predict] <- recall
  
  index.predict <- index.predict + 1
  #print(confusion_matrix[1,2]/sum(confusion_matrix))
}
Optimize_threshold <- data.frame(Prob_Threshold,missed_churns,predicted_churns,FalsePositive,TruePositive,F1_score.Prob)

# Plot F1-Score based on threshold cutoff
# calculate max f1-score
max_index <- which.max(F1_score.Prob)
max_Prob_Threshold <- Prob_Threshold[max_index]
max_precision <- precision.Prob[max_index]
max_recall <- recall.Prob[max_index]
max_f1_score <- F1_score.Prob[max_index]

plot(Prob_Threshold, F1_score.Prob, type = "l", xlab = "Probability threshold cutoffs",
     ylab = "F1-score", main = "F1 score - Probability threshold cutoffs", col = "blue")
abline(v = max_Prob_Threshold, col = "red")
text(max_Prob_Threshold, max_f1_score, labels = paste("Max F1-score:",
                                                      round(max_f1_score, 2), "\nThreshold cutoff:", round(max_Prob_Threshold, 2)), pos = 1)

# Plot Missed-Predicted Churn 
#par("mar")
#plot(1:30)
plot(predicted_churns,missed_churns, type = "l", xlab = "Predicted Churns Ratio",
     ylab = "Missed Churns Ratio", main = "Missed-Predicted Churn",col = "blue")
mtext("Numbers on plot are probability threshold cutoffs for predicting churn", side = 3, line = 0, cex = 1)
labels <- Prob_Threshold[seq(1, length(Prob_Threshold), by = 10)]  # Select every 5th value
text(predicted_churns[seq(1, length(Prob_Threshold), by = 10)], missed_churns[seq(1, length(Prob_Threshold), by = 10)], labels = labels, pos = 3)

# Plot False Positive Rate vs True Positive Rate (ROC Curve)
#plot(FalsePositive, TruePositive, type = "l", xlab = "False Positive Rate", ylab = "True Positive Rate", main = "ROC Curve")
#------------------------------------------------------------
detach(telco)
#### MAP ######--------------------
# Load required library
library(leaflet)

telco1 <- read_csv("Telco_customer_churn.csv")
library(janitor)
library(dplyr)

telco1 <- telco1%>% clean_names()
names(telco1)



# Read status.csv file
status <- read_csv("Telco_customer_churn_status.csv")
status <- status%>% clean_names()
names(status)

telco1 <- merge(telco1, status[, c("customer_id", "churn_category")], by = "customer_id", all.x = TRUE)


telco1 <- na.omit(telco1)
dim(telco1)

attach(telco1)
telco_subset <- data.frame(latitude,longitude,churn_value,churn_category)


# Assuming your dataset is named 'data' with columns 'lat', 'long', and 'churn'


# Create a leaflet map
map <- leaflet(telco_subset) %>%
  addTiles() %>%
  addCircleMarkers(
    ~longitude, ~latitude,
    color = ~case_when(
      churn_category == "Competitor" ~ "red",
      churn_category == "Dissatisfaction" ~ "blue",
      churn_category == "Attitude" ~ "green",
      churn_category == "Price" ~ "orange",
      TRUE ~ "gray"  # Default color for other categories
    ),
    radius = 2,  # Adjust the size of markers as needed
    popup = ~paste("Churn:", churn_value)  # Show churn status in popup
  )

map

# Create a color palette for churn categories
churn_colors <- c("Competitor" = "red",
                  "Dissatisfaction" = "blue",
                  "Attitude" = "green",
                  "Price" = "orange",
                  "Other" = "gray")

# Create a legend control
legend_values <- names(churn_colors)
legend_colors <- unname(churn_colors)

map <- addLegend(map = map,
                 position = "bottomleft",
                 colors = legend_colors,
                 labels = legend_values,
                 title = "Churn Category")
map

#----------------------Model Evaluation-------------------------
round(precision.tree[2],2)
round(recall.tree[2],2)
round(F1_score.tree[2],2)

round(precision.rf50[2],2)
round(recall.rf50[2],2)
round(F1_score.rf50[2],2)

round(max_precision,2) 
round(max_recall,2) 
round(max_f1_score,2) 

evaluation.data <- data.frame(
  Model = c("Decision Tree", "RF 50% threshold", "RF 33% threshold"),
  Precision = c(round(precision.tree[2], 2), round(precision.rf50[2], 2), round(max_precision, 2)),
  Recall = c(round(recall.tree[2], 2), round(recall.rf50[2], 2), round(max_recall, 2)),
  F1_score = c(round(F1_score.tree[2], 2), round(F1_score.rf50[2], 2), round(max_f1_score, 2))
)
print(evaluation.data)
