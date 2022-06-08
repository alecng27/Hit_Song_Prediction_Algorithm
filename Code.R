# Project Codes for reproducibility

# Loading Libraries
library(tidyverse)
library(caret)
library(recipes)
library(earth)
library(ipred)
library(e1071)
library(ranger)
library(rpart)
library(rpart.plot)
library(skimr)
library(corrplot)
library(caret)
library(stringr)
library(stopwords)
library(snakecase)
library(gridExtra)
library(dplyr)
library(kableExtra)

######### Importing data ######### 
Spotify <- read_csv("Datasets/Spotify-2000.csv")

######### Quick cleaning ######### 
Spotify <- Spotify %>%
  # Remove the index column
  select(-c(Index)) %>%
  # Convert character in to factor
  mutate_if(is.character,as.factor) %>%
  # Rename with cleaner names
  rename(Genre = `Top Genre`,
         Length = `Length (Duration)`,
         BPM = `Beats Per Minute (BPM)`,
         Loudness = `Loudness (dB)`)

######### Combine genre ######### 
combine_sheet <-read_csv("Datasets/CMSC (Genre).csv") # Pre-made combine sheet

######### Mutate dataset to use the combined genres ######### 
Spotify <- Spotify %>%
  mutate(Combined_Genre = case_when(
    Genre %in% combine_sheet$`Rock/Metal` ~ "Rock_Metal",
    Genre %in% combine_sheet$Alternative ~ "Alternative",
    Genre %in% combine_sheet$Punk ~ "Punk",
    Genre %in% combine_sheet$`Folk/Indie` ~ "Folk_Indie",
    Genre %in% combine_sheet$Classic ~ "Classic",
    Genre %in% combine_sheet$Soul ~ "Soul",
    Genre %in% combine_sheet$Dance ~ "Dance",
    Genre %in% combine_sheet$Jazz ~ "Jazz",
    Genre %in% combine_sheet$`Singer-Songwriter` ~ "Singer_Songwriter",
    Genre %in% combine_sheet$`*Country` ~ "Country",
    Genre %in% combine_sheet$`(Hip)Pop` ~ "Hip_Pop",
    TRUE ~ "Other"
  )) %>%
  mutate(Combined_Genre = as.factor(Combined_Genre))

########## Create a new variable, length of title ######### 
Spotify <- Spotify %>%
  mutate(Title_Word_Count = str_count(Spotify$Title, pattern = "\\w+"))

# Split data
set.seed(888)
# Split data
Spotify_index <- createDataPartition(Spotify$Popularity, p = 0.8, list = FALSE)
Spotify_train <- Spotify[Spotify_index,]
Spotify_test <- Spotify[-Spotify_index,] 

############# Create 25 most common words ############# 
# Firstly, we will have to lower case titles
Spotify_train <- Spotify_train %>% mutate(Title = str_to_lower(Title))
Spotify_test  <- Spotify_test  %>% mutate(Title = str_to_lower(Title))

# Secondly, we find which words do we want to create a dummy for, from TRAINING
word_count <- data.frame(matrix(ncol = 2, nrow = 0))
colnames(word_count) <- c('word', 'count')
black_list <- stopwords()
for (title in strsplit(Spotify_train$Title, "[- ,\\(\\)\"]")) { # Split on common separators 
  for (word in title) {
    if (!(word %in% black_list)) { # Filter out numbers and common words, just common library for now
      if (word %in% word_count$word) {
        word_count$count[which(word_count$word == word)] <- word_count$count[which(word_count$word == word)] + 1
      } else {
        word_count <- rbind(word_count, data.frame(word = word, count = 1))
      }
    }
  }
}

word_count <- arrange(word_count, desc(count)) # Arrange by count
word_count <- word_count[-c(1),] # Remove most common: white space
significant_words <- head(word_count,25)$word # Just take most common 25 for now, can tune later

# Thirdly,for each of the word in the list, we create a dummy variable

# Helper function to check if word is in a title (have to be custom made since the split above is also custom)
check_word <- function(title, word) {
  word_list <- strsplit(title, "[- ,\\(\\)\"]")[[1]]
  return(word %in% word_list)
}
check_word_expanded <- function(word, titles) {
  sapply(titles, check_word, word = word)
}

word_mutate <- function(data, word_list) {
  new_data <- data
  for (word in word_list) {
    col_name <- to_any_case(paste("contains", word),case = "snake")
    new_data <- new_data %>% 
      mutate("{col_name}" := ifelse(check_word_expanded(word, Title), 1, 0))
  }
  return(new_data)
}

# Apply funciton onto training and testing datasets
Spotify_train <- word_mutate(Spotify_train, significant_words)
Spotify_test <- word_mutate(Spotify_test, significant_words)

############# Create 25 most repeated artists ############# 
# First create a list of 25 most repeated artists from TRAINING data set
top_25_artists <- Spotify_train %>% group_by(Artist) %>%
  summarize(n = n()) %>%
  arrange(desc(n)) %>%
  head(25) # Top 25 for now, tune later

# Create function to create dummy variable to each artist given a list
artist_mutate <- function(data, artist_list) {
  new_data <- data
  for (artist in artist_list) {
    col_name <- to_any_case(paste("By", artist),case = "snake")
    new_data <- new_data %>% 
      dplyr::mutate("{col_name}" := ifelse(artist == Artist, 1, 0))
  }
  return(new_data)
}

# Apply the function on both train and test dataset
Spotify_train <- artist_mutate(Spotify_train, top_25_artists$Artist)
Spotify_test <- artist_mutate(Spotify_test, top_25_artists$Artist)

# We deselect Title, Artist, and Genre
Spotify_train <- select(Spotify_train, !c(Title, Artist, Genre))
Spotify_test <- select(Spotify_test, !c(Title, Artist, Genre))

# preprocessing
spotify_recipe <- recipe(Popularity ~. , data = Spotify_train)   # create the recipe for blueprint

# spotify_recipe$var_info   # check the types and roles of variables
# nearZeroVar(Spotify_train, saveMetrics = TRUE)   # The majority of the custom dummies are nzv, but that is to be expected for premade dummies

blueprint <- spotify_recipe %>%
  step_center(all_numeric_predictors(), -starts_with("by"), -starts_with("contains")) %>%   # center all numeric features except response and premade dummies
  step_scale(all_numeric_predictors(), -starts_with("by"), -starts_with("contains")) %>%    # scale all numeric features except response and premade dummies
  step_dummy(all_nominal_predictors(), one_hot = FALSE) # create dummy variables for nominal categorical features

# blueprint
prepare <- prep(blueprint, training = Spotify_train)   # estimate feature engineering parameters from training set
baked_train <- bake(prepare, new_data = Spotify_train)  # apply blueprint to training set
baked_test <- bake(prepare, new_data = Spotify_test)    # apply blueprint to test set

set.seed(888)
# We choose the CV setting of 5 fold and 5 repeats
cv_specs <- trainControl(method = "repeatedcv", number = 5, repeats = 5)

set.seed(888)
################### MLR ###################
mlr_cv <- train(blueprint,
                data = Spotify_train, 
                method = "lm",
                trControl = cv_specs,
                metric = "RMSE")
min(mlr_cv$results$RMSE) # MLR CV RMSE

set.seed(888)
################### KNN ###################
k_grid <- expand.grid(k = seq(1, 15, 1))   # for KNN regression

knn_cv <- train(blueprint,
                data = Spotify_train, 
                method = "knn",
                trControl = cv_specs,
                tuneGrid = k_grid,
                metric = "RMSE")
min(knn_cv$results$RMSE) # KNN CV RMSE

set.seed(888)
################### Ridge Regression ###################
lambda_grid <- 10^seq(-3, 3, length = 100)   # grid of lambda values to search over

ridge_cv <- train(blueprint,
                  data = Spotify_train,
                  method = "glmnet",   # for ridge regression
                  trControl = cv_specs,
                  tuneGrid = expand.grid(alpha = 0, lambda = lambda_grid),  # alpha = 0 implements ridge regression
                  metric = "RMSE")
min(ridge_cv$results$RMSE) # Ridge CV RMSE

g1 <- ggplot(ridge_cv)   # lambda vs. RMSE plot
g2 <- ggplot(ridge_cv) + xlim(c(0,2))    # a closer look at lambda vs. RMSE plot
grid.arrange(g1, g2, ncol = 2)

set.seed(888)
################### LASSO Regression ###################
lasso_cv <- train(blueprint,
                  data = Spotify_train,
                  method = "glmnet",   # for lasso
                  trControl = cv_specs,
                  tuneGrid = expand.grid(alpha = 1, lambda = lambda_grid),  # alpha = 1 implements lasso
                  metric = "RMSE")
min(lasso_cv$results$RMSE) # Lasso CV RMSE

g3 <- ggplot(lasso_cv)   # lambda vs. RMSE plot
g4 <- ggplot(lasso_cv) + xlim(c(0,2))    # a closer look at lambda vs. RMSE plot
grid.arrange(g3, g4, ncol = 2)

set.seed(888)
################### MARS ################### 
param_grid_mars <- expand.grid(degree = 1:3, nprune = seq(1, 100, length.out = 10))   # for MARS

mars_cv <- train(blueprint,
                 data = Spotify_train,
                 method = "earth",
                 trControl = cv_specs,
                 tuneGrid = param_grid_mars,
                 metric = "RMSE")
min(mars_cv$results$RMSE) # MARS CV RMSE

set.seed(888)
################### Regression Tree ################### 
tree_cv <- train(blueprint,
                 data = Spotify_train,
                 method = "rpart",
                 trControl = cv_specs,
                 tuneLength = 20,
                 metric = "RMSE")
min(tree_cv$results$RMSE) # Regression tree CV RMSE

rpart.plot(tree_cv$finalModel)

set.seed(888)
################### Bagged ################### 
# Tutorial https://bradleyboehmke.github.io/HOML/bagging.html section 10.4
bag_cv <- train(blueprint,
                data = Spotify_train, 
                method = "treebag",
                trControl = cv_specs,
                nbagg = 500,
                control = rpart.control(minsplit = 2, cp = 0, xval = 0),
                metric = "RMSE")
library(dplyr) # Prevent plyr overwrite dplyr
min(bag_cv$results$RMSE) # Bagged CV RMSE

# Top 10 most important variables
varImp(bag_cv$finalModel) %>% arrange(desc(Overall)) %>% head(10)

set.seed(888)
################### Random Forests ################### 
param_grid_rf <- expand.grid(mtry = seq(1, ncol(baked_train) - 1, 1), # for random forests # sequence of 1 to number of predictors
                             splitrule = "variance",  # "gini" for classification
                             min.node.size = 2)       # for each tree

rf_cv <- train(blueprint,
               data = Spotify_train,
               method = "ranger",
               trControl = cv_specs,
               tuneGrid = param_grid_rf,
               importance = "permutation", # needed to use varImp, check https://stackoverflow.com/questions/37279964/variable-importance-with-ranger
               metric = "RMSE")
min(rf_cv$results$RMSE) # Random forest CV RMSE

# Top 10 most important variables
varImp(rf_cv)$importance %>% arrange(desc(Overall)) %>% head(10)

# fit final model
rffit <- ranger(Popularity ~ .,
                data = baked_train,
                num.trees = 500,
                mtry = rf_cv$bestTune$mtry,
                splitrule = "variance",   # use "gini" for classificaton
                min.node.size = 2,
                importance = "impurity")

# variable importance
data.frame(Overall = rffit$variable.importance) %>% arrange(desc(Overall)) %>% head(10)

# R-Squared of the final model
rffit$r.squared

preds_rf <- predict(rffit, data = baked_test, type = "response")  # predictions on test set
pred_error_est_rffit <- sqrt(mean((preds_rf$predictions - baked_test$Popularity)^2))  # test set RMSE
pred_error_est_rffit









