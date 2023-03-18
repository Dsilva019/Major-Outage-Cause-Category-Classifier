# Major Outage Cause Category Classifier
This is a project for DSC 80 at UCSD. Our exploratory data analysis on this dataset can be found <a href="https://dsilva019.github.io/EDA-of-Major-Outages/"> here </a>.


# Introduction

For this project using the outages data set I want to be able to predict the category cause that resulted in the major outage. Since I am trying to predict the category of the cause of the major outage, the type of problem I have is a classification problem. As a result, I will be predicting the category cause of future major outages using a Decision Tree Classifier (DTC). My response variable for my regression problem is the Category Cause. I will be accessing the quality of my model using the accuracy metric of the DTC to see how well my model predicts the category cause of an outage. To train my model I will be using a data set containing major outages reported by different states in the United States from January 2000-July 2016. This data set contains 1534 rows and 55 columns.





# Data Cleaning

In my data cleaning process, I first looked at the raw data set to assess what steps needed to be done. I looked for any unnecessary rows and columns, checked the column names, and anything else that looked out of the ordinary for a data set. In my specific case, I noticed there were a couple of columns and rows that were unnecessary and the column names were incorrect. So I dropped said columns and rows, set the columns to their proper respective names, and reset the index of the data frame. Lastly, I ensured the data types of the columns were the best possible type that allowed me to properly analyze the data frame.  

# Outages DataFrame

|   YEAR |   MONTH | U.S._STATE   | POSTAL.CODE   | NERC.REGION   | CLIMATE.REGION     |   ANOMALY.LEVEL | CLIMATE.CATEGORY   | OUTAGE.START.DATE   | OUTAGE.START.TIME   | OUTAGE.RESTORATION.DATE   | OUTAGE.RESTORATION.TIME   | CAUSE.CATEGORY     | CAUSE.CATEGORY.DETAIL   |   HURRICANE.NAMES |   OUTAGE.DURATION |   DEMAND.LOSS.MW |   CUSTOMERS.AFFECTED |   RES.PRICE |   COM.PRICE |   IND.PRICE |   TOTAL.PRICE |   RES.SALES |   COM.SALES |   IND.SALES |   TOTAL.SALES |   RES.PERCEN |   COM.PERCEN |   IND.PERCEN |   RES.CUSTOMERS |   COM.CUSTOMERS |   IND.CUSTOMERS |   TOTAL.CUSTOMERS |   RES.CUST.PCT |   COM.CUST.PCT |   IND.CUST.PCT |   PC.REALGSP.STATE |   PC.REALGSP.USA |   PC.REALGSP.REL |   PC.REALGSP.CHANGE |   UTIL.REALGSP |   TOTAL.REALGSP |   UTIL.CONTRI |   PI.UTIL.OFUSA |   POPULATION |   POPPCT_URBAN |   POPPCT_UC |   POPDEN_URBAN |   POPDEN_UC |   POPDEN_RURAL |   AREAPCT_URBAN |   AREAPCT_UC |   PCT_LAND |   PCT_WATER_TOT |   PCT_WATER_INLAND |
|-------:|--------:|:-------------|:--------------|:--------------|:-------------------|----------------:|:-------------------|:--------------------|:--------------------|:--------------------------|:--------------------------|:-------------------|:------------------------|------------------:|------------------:|-----------------:|---------------------:|------------:|------------:|------------:|--------------:|------------:|------------:|------------:|--------------:|-------------:|-------------:|-------------:|----------------:|----------------:|----------------:|------------------:|---------------:|---------------:|---------------:|-------------------:|-----------------:|-----------------:|--------------------:|---------------:|----------------:|--------------:|----------------:|-------------:|---------------:|------------:|---------------:|------------:|---------------:|----------------:|-------------:|-----------:|----------------:|-------------------:|
|   2011 |       7 | Minnesota    | MN            | MRO           | East North Central |            -0.3 | normal             | 2011-07-01 00:00:00 | 17:00:00            | 2011-07-03 00:00:00       | 20:00:00                  | severe weather     | nan                     |               nan |              3060 |              nan |                70000 |       11.6  |        9.18 |        6.81 |          9.28 | 2.33292e+06 | 2.11477e+06 | 2.11329e+06 |   6.56252e+06 |      35.5491 |      32.225  |      32.2024 |         2308736 |          276286 |           10673 |           2595696 |        88.9448 |        10.644  |       0.411181 |              51268 |            47586 |          1.07738 |                 1.6 |           4802 |          274182 |       1.75139 |             2.2 |      5348119 |          73.27 |       15.28 |           2279 |      1700.5 |           18.2 |            2.14 |          0.6 |    91.5927 |         8.40733 |            5.47874 |
|   2014 |       5 | Minnesota    | MN            | MRO           | East North Central |            -0.1 | normal             | 2014-05-11 00:00:00 | 18:38:00            | 2014-05-11 00:00:00       | 18:39:00                  | intentional attack | vandalism               |               nan |                 1 |              nan |                  nan |       12.12 |        9.71 |        6.49 |          9.28 | 1.58699e+06 | 1.80776e+06 | 1.88793e+06 |   5.28423e+06 |      30.0325 |      34.2104 |      35.7276 |         2345860 |          284978 |            9898 |           2640737 |        88.8335 |        10.7916 |       0.37482  |              53499 |            49091 |          1.08979 |                 1.9 |           5226 |          291955 |       1.79    |             2.2 |      5457125 |          73.27 |       15.28 |           2279 |      1700.5 |           18.2 |            2.14 |          0.6 |    91.5927 |         8.40733 |            5.47874 |
|   2010 |      10 | Minnesota    | MN            | MRO           | East North Central |            -1.5 | cold               | 2010-10-26 00:00:00 | 20:00:00            | 2010-10-28 00:00:00       | 22:00:00                  | severe weather     | heavy wind              |               nan |              3000 |              nan |                70000 |       10.87 |        8.19 |        6.07 |          8.15 | 1.46729e+06 | 1.80168e+06 | 1.9513e+06  |   5.22212e+06 |      28.0977 |      34.501  |      37.366  |         2300291 |          276463 |           10150 |           2586905 |        88.9206 |        10.687  |       0.392361 |              50447 |            47287 |          1.06683 |                 2.7 |           4571 |          267895 |       1.70627 |             2.1 |      5310903 |          73.27 |       15.28 |           2279 |      1700.5 |           18.2 |            2.14 |          0.6 |    91.5927 |         8.40733 |            5.47874 |
|   2012 |       6 | Minnesota    | MN            | MRO           | East North Central |            -0.1 | normal             | 2012-06-19 00:00:00 | 04:30:00            | 2012-06-20 00:00:00       | 23:00:00                  | severe weather     | thunderstorm            |               nan |              2550 |              nan |                68200 |       11.79 |        9.25 |        6.71 |          9.19 | 1.85152e+06 | 1.94117e+06 | 1.99303e+06 |   5.78706e+06 |      31.9941 |      33.5433 |      34.4393 |         2317336 |          278466 |           11010 |           2606813 |        88.8954 |        10.6822 |       0.422355 |              51598 |            48156 |          1.07148 |                 0.6 |           5364 |          277627 |       1.93209 |             2.2 |      5380443 |          73.27 |       15.28 |           2279 |      1700.5 |           18.2 |            2.14 |          0.6 |    91.5927 |         8.40733 |            5.47874 |
|   2015 |       7 | Minnesota    | MN            | MRO           | East North Central |             1.2 | warm               | 2015-07-18 00:00:00 | 02:00:00            | 2015-07-19 00:00:00       | 07:00:00                  | severe weather     | nan                     |               nan |              1740 |              250 |               250000 |       13.07 |       10.16 |        7.74 |         10.43 | 2.02888e+06 | 2.16161e+06 | 1.77794e+06 |   5.97034e+06 |      33.9826 |      36.2059 |      29.7795 |         2374674 |          289044 |            9812 |           2673531 |        88.8216 |        10.8113 |       0.367005 |              54431 |            49844 |          1.09203 |                 1.7 |           4873 |          292023 |       1.6687  |             2.2 |      5489594 |          73.27 |       15.28 |           2279 |      1700.5 |           18.2 |            2.14 |          0.6 |    91.5927 |         8.40733 |            5.47874 |

# baseline Model


### Features
Since I will be building a Decision Tree Classifier, I will pick 2 features that I believe will be good variables to predict the number of affected in future major outages. The features I want to use to build my model are 'MONTH' and 'NERC.REGION'. I chose these features because after performing a quick exploratory data analysis, noticed a significant amount of the cause category of major outages were due to severe weather. So I chose two variables that I believe reflect the my findings. In my features I have two nominal categorical features.

|   MONTH | NERC.REGION   |
|--------:|:--------------|
|       7 | MRO           |
|       5 | MRO           |
|      10 | MRO           |
|       6 | MRO           |
|       7 | MRO           |
|      11 | MRO           |
|       7 | MRO           |
|       6 | MRO           |
|       3 | MRO           |
|       6 | MRO           |

# Feature Engineering & Preprocessing

My categorical features needs to be engineered so that it can be used in my Decision Tree Classifier model. Since all my categorical features are nominal I will use one hot encode to engineer them from a categorical feature into numerical feature so that my model can use it.

#### Baseline Pipeline

To build my model I will be using a pipeline to preprocess the categorical features into numerical ones and then using all my features to train a decision tree classifier. I will also be splitting my data into to sections training and testing data. My model will be trained on purely the training data but I will access the accuracy of model using both training and testing data to compare the two results.

### Summary of Results

My current baseline model has an accuracy of 0.605 on training data and a score of 0.536 on testing data. This tells me while my model is able to predict the outcome of training data not that well and on new never seen data my model performs even worse. This tells me model is slightly overfitted with the training data and needs more features since it only accurate 53.6% percent of the time when tested on new data.

## Final model

### New Features

As previously stated my to improve my baseline model I needed more features ones that give more insight to what the category cause of the major outage was. I did some more exploratory data analysis and found that the second and third highest counts of category causes of major outages were intentional attack and system operability disruption. These category causes are more general than severe weather so I looked for more features that could reflect that. So the new features I chose were, 'CLIMATE.REGION', 'POPULATION', 'HURRICANE.NAMES', 'AREAPCT_URBAN', and 'PCT_WATER_TOT'. For the features 'U.S._STATE', 'POPULATION', and 'AREAPCT_URBAN' since these features reflect the demographics of a the people affected by the major outages I believe would help to explain the high number of major outages due to intentional attack and system operability disruption. As for the features 'CLIMATE.REGION', 'HURRICANE.NAMES', 'AREAPCT_URBAN', and 'PCT_WATER_TOT' since severe weather is still the most common reason to why major outages occurred I believe this would also help to provide my model better information on weather when to predict a major outages occurred due to severe weather or not. Since it gives insight to weather the outage was due to a hurricane, the climate of the region that the major occurred, and the amount of water in the state that the major outage occurred these gives more data about the weather conditions I hope it will allow my model to make better predictions with the extra information.


### New Preprocessing line
Custom Function Transformer that binarizes the Hurricane Names.
New Preprocessing Line that one hot encodes 'MONTH' ,'U.S._STATE','NERC.REGION', 'CLIMATE.REGION', and turns the Population into percentiles. lastly it leaves 'AREAPCT_URBAN', 'PCT_WATER_TOT' as is since there values are in percentages that range from 0-99%


### Decision Tree Classifier Hyperparameter Fine Tuner

To further improve my model I decided I want to fine tune the hyper parameters of the Decision Tree Classifier. Those being the max depth and the minimum sample split. The reasons why I chose to tune the max depth is because I want to make my tree more expressive since there are multiple cause categories I want my tree to be a bit more complex. Secondly, the reason why I chose to fine tune the minimum sample split of the tree is because since it is not guaranteed that my tree will be built symmetrically. As a result, to optimize generalization performance by increasing the number of minimum sample splits it allows some branches to grow deeper than others. This will allow my tree more splits to better classify the samples.

## Summary of Final Model and results

### Final Model Breakdown

In conclusion, the final model I chose was a Decision tree Classifier. The features I chose for my model were 'MONTH', 'U.S._STATE' ,'NERC.REGION', 'CLIMATE.REGION', 'POPULATION', 'HURRICANE.NAMES', 'AREAPCT_URBAN', and 'PCT_WATER_TOT'. Five of those features were nominal categorical features, while the three of them were numerical features. I one hot encoded all four of the nominal categorical features to turn them into numerical ones. I binarized 'HURRICANE.NAMES' using a custom function transformer I made to engineer the its values into 1's and 0's. As for the numerical values turned the 'POPULATION' values into quantiles and left 'AREAPCT_URBAN', and 'PCT_WATER_TOT' as is. As for the hyperparameters of the Decision Tree Classifier, I ended up choosing a max depth of 6 and minimum sample split of 5. The way I did this was using a double for loop that would test every single combination of a max depth range of 1-200 and minimum sample split range of 2-30. In each iteration I fitted the model with training data and recorded its accuracy on the testing data. Then I chose the model with the greatest accuracy with the testing data.

### Results Breakdown

In conclusion, the final model I chose was a Decision tree Classifier. The features I chose for my model were 'MONTH', 'U.S._STATE' ,'NERC.REGION', 'CLIMATE.REGION', 'POPULATION', 'HURRICANE.NAMES', 'AREAPCT_URBAN', and 'PCT_WATER_TOT'. Five of those features were nominal categorical features, while the three of them were numerical features. I one hot encoded all four of the nominal categorical features to turn them into numerical ones. I binarized 'HURRICANE.NAMES' using a custom function transformer I made to engineer the its values into 1's and 0's. As for the numerical values turned the 'POPULATION' values into quantiles and left 'AREAPCT_URBAN', and 'PCT_WATER_TOT' as is. As for the hyperparameters of the Decision Tree Classifier, I ended up choosing a max depth of 6 and minimum sample split of 5. The way I did this was using a double for loop that would test every single combination of a max depth range of 1-200 and minimum sample split range of 2-30. In each iteration I fitted the model with training data and recorded its accuracy on the testing data. Then I chose the model with the greatest accuracy with the testing data.

# Fairness analysis

To access the fairness of my final model, I want to see whether my model is fair when predicting the cause category of major outages between low and high population states. I will continue to use accuracy as my evaluation metric to conduct my fairness analysis. Since there is not an exact definition of low and high population states I define my own definitions here. Low population states are states with a Population quantile of three or lower, high population are states with Population Quantiles greater than 3. The test statistic I will be using will be the absolute difference in accuracy. I will choose a significance level of 0.05 as a cut-off for my p-value, since a p-value smaller than 0.05 indicates strong evidence against my null hypothesis.


### Hypotheses:

#### Null Hypothesis: The classifier's accuracy is the same for both low population states and high population states, and any differences are due to chance.

#### Alternative Hypothesis: There is a difference in accuracy for low population states and high population states

#### Observed Absolute Difference in Accuracy: 0.03581703572005823

### Set Up To begin my fairness analysis, I first need to create a new column that contains the Population Quantiles of the state that the major outages occurred in.

### Summary of Results

Since, the p-value is greater than the significance level, 0.47 > 0.05, we fail to reject the null hypothesis. There is not enough evidence to suggest that there is a difference in accuracy between low population states and high population states.
