library(AppliedPredictiveModeling)

# Load the dataset
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
data("segmentationOriginal")

segData <- subset(segmentationOriginal, Case == "Train")

cellID <- segData$Cell
class <- segData$Class
case <- segData$Case
# Now remove the columns
segData <- segData[, -(1:3)]

statusColNum <- grep("Status", names(segData))

segData <- segData[, -statusColNum]

# Transformations
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# When features exhibit significant skewness.  e1071 package calculates the 
# sample skewness statistic for each predictor.

library(e1071)
# For one predictor:
skewness(segData$AngleCh1)
 
# Since all the predictors are  numeric columns, the apply function can
# be used to compute the skewness across columns.
skewValues <- apply(segData, 2, skewness)
head(skewValues)

# Using these values as a guide, the variables can be prioritized for visualizing
# the distribution.  The basic R function 'hist' or the 'histogram' function
# in the lattice can be used to assess the shape of the distribution.

# Determine the type of transformation that should be used - MASS package
# contains the 'boxcox' function. This function estimates lambda, it does
# not create the transformed variable(s).  A 'caret' function, 'BoxCoxTrans' 
# can find the appropriate tranformation and apply them to the new data.

library(caret)
Ch1AreaTrans <- BoxCoxTrans(segData$AreaCh1)

# The original data
head(segData$AreaCh1)
# After transformation
predict(Ch1AreaTrans, head(segData$AreaCh1))
# checking on the first data point
(819^(-0.9)-1)/(-0.9)

# Another 'caret' function, 'preProcess' applies this transformation to a set
# of predictors.  The base R function 'prcomp' can be used for PCA.  
# below - data are centered and scaled prior to PCA.

pcaObject <- prcomp(segData,
                    center = TRUE, scale. = TRUE)

# Calculate the cumulative percentage of variance which each component
# accounts for.

percentVariance <- pcaObject$sd^2/sum(pcaObject$sd^2)*100

# The transformed values are stored in 'pcaObject' as a sub-object
# called x: 
head(pcaObject$x[, 1:5])

# Another sub-object called 'rotation' stores the variable loadings,
# where rows correspond to predictor variables and columns are 
# associated with the components.

head(pcaObject$rotation[, 1:3])

# To administer a series of transformation to multiple data sets,
# the 'caret' class 'preProcess' has the ability to transform, center, 
# scale, or impute values, as well as apply the spatial sign transformation 
# and feature extraction.  The function calculates the required 
# quantities for the transformation.  Afer calling the 'preProcess'
# function, the 'predict' method applies the results to a set of data.
# For example, to Box-Cox transform, center, and scale the data, then
# execute PCA for signal extraction:

trans <- preProcess(segData, 
                    method = c("BoxCox", "center", "scale", "pca"))

# Apply the transformations:
transformed <- predict(trans, segData)
# These values are different than the previous PCA components
# since they were transformed prior to PCA.
head(transformed[, 1:5])

# The order in which the possible transformation are applied is transformation,
# center, scaling, imputation, feature extraction, and then spatial sign.

# Filtering
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# To filter for near zero predictors, the 'caret' package function 
# 'nearZeroVar' will return the column numbers of any predictors that fulfil
# the conditions.  For cell segmentationi data, there are not problematic 
# predictors.

nearZeroVar(segData)

# When predictors should be removed, a vector of integers is returned
# that indicates which columns should be removed.

# Similarly, to filter on between-predictor correlations, the 'cor'
# functioni can calculate teh correlations between predictor variables:

correlations <- cor(segData)
dim(correlations)
correlations[1:4, 1:4]

# to visually examine the correlation sturcute of the data, the 
# 'corrplot' package contains an excellent function of the same 
# name.  The function has many options includeing one that will 
# reorder the variables in a way that reveals clusters of 
# highly correlated predictors.  

library(corrplot)
corrplot(correlations, order = 'hclust')

# To filter based on correlations, the 'findCorrelation' function
# will apply the alogrithm.  For a given threshold of pairwise correlations, the 
# function returns column numers denoting the predictors that are recommended for 
# deletion.

highCorr <- findCorrelation(correlations, cutoff = .75)
length(highCorr)
head(highCorr)

filterSegData <- segData[, -highCorr]

# Creating Dummy Variables
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# To illustrate this we are using a subset of the 'cars' data set in the 'caret'
# package.  2005 Kelly Blue Book resale data for 804 GM cars.  The object of the 
# model was to predict the price of the car based on known characteristics.  
# This focuses on the price, mileage, and car type for a subset of vehicles.

data("cars")
df_cars <- subset(cars, Price >= 0)

# select variables price, mileage, and car type
myvars <- c("Price", "Mileage", "convertible", "coupe", "hatchback", "sedan", "wagon")
carSubset <- df_cars[myvars]
head(carSubset)

carSubset$convertible[carSubset$convertible == 1] <- 'convertible'
carSubset$coupe[carSubset$coupe == 1] <- 'coupe'
carSubset$hatchback[carSubset$hatchback == 1] <- 'hatchback'
carSubset$sedan[carSubset$sedan == 1] <- 'sedan'
carSubset$wagon[carSubset$wagon == 1] <- 'wagon'

library(dplyr)

x <- carSubset %>%
  mutate(Type = case_when(convertible == 'convertible' ~ 'convertible',
                          coupe == 'coupe' ~ 'coupe',
                          hatchback == 'hatchback' ~ 'hatchback',
                          sedan == 'sedan' ~ 'sedan',
                          wagon == 'wagon' ~ 'wagon'))

# select variables price, mileage, and car type
myvars <- c("Price", "Mileage", "Type")
carSubset <- x[myvars]
head(carSubset)

levels(carSubset$Type)

# to model the price as a function of mileage and type of car, use
# function 'dummyVars' to determine encoding for the predictors.

# first model assumes that the price can be modeled as a simple
# additive function of the mileage and type.

simpleMod <- dummyVars(~Mileage + Type, 
                       data = carSubset,
                       ## Remove the variable name from the
                       ## column name
                       levelsOnly = TRUE)

# To generate the dummy bariables for the training set or any
# new samples, the 'predict' method is used in conjunction
# with the 'dummyVars' object:

# assumes that the effect of mileage is the same on every type
# of car.
predict(simpleMod, head(carSubset))

# A more advanced model assumes that there is a joint effect of mileage
# and car type.  This type of effect is referred to as a 'joint' effect
# of mileage and car type.  This type of effect is referred to as an 
# interaction.

withInteractions <- dummyVars(~Mileage + Type + Mileage:Type,
                              data = carSubset,
                              levelsOnly = TRUE)

predict(withInteractions, head(carSubset))
