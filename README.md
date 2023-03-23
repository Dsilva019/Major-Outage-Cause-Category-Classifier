# Major Outage Cause Category Classifier
<small> <i>By Diego Silva (d1silva@ucsd.edu). My exploratory data analysis on this dataset can be found <a href="https://dsilva019.github.io/EDA-of-Major-Outages/"> here</a>.</i> </small>


# <u>Introduction</u>

For this project, I will be using a data set containing major outages reported by different states in the United States from January 2000-July 2016. This data set contains 1534 rows and 55 columns. My goal for this project is to be able to predict the category cause that resulted in a major outage. Since I am trying to predict the category of the cause of the major outage, the type of problem I have is a multiclass classification problem. As a result, I will be predicting the category cause of future major outages using a Decision Tree Classifier (DTC). My response variable for my multiclass classification problem is the Category Cause. I decided to predict the Category Cause of major outages because I wanted to see if it is possible based on a given set of conditions to predict the cause of a major outage that occurred. The type of data I will have access will be of the following geographical data, time data, weather data, and demographics. Specifically, the data I will be using are the Month, U.S State, NERC Region, Climate Region, Hurricane Name, Population number, the Percentage of the Urban Population of the total population in the U.S State. Since I am trying to predict the category cause for a major outage that already occurred, the data previously would already be recorded and available for me to use. I will be accessing the quality of my model using the accuracy metric of the DTC to see how well my model predicts the category cause of a major outage. I chose accuracy as my evaluation metric because I want to see how many predictions my model got correct overall, I am not worried about false positives or negatives of my model and for the scope of my problem precision or recall do not have much importance since there are no negative outcomes of any false positives or negatives my model might make. For the scope of my problem, overall accuracy is more important than knowing the false predictions.





#### Data Cleaning

In my data cleaning process, I first looked at the raw data set to assess what steps needed to be done. I looked for any unnecessary rows and columns, checked the column names, and anything else that looked out of the ordinary for a data set. In my specific case, I noticed there were a couple of columns and rows that were unnecessary and the column names were incorrect. So I dropped said columns and rows, set the columns to their proper respective names, and reset the index of the data frame. Lastly, I ensured the data types of the columns were the best possible type which allowed me to properly analyze the data frame. Lastly, I combined columns 'OUTAGE.START.DATE' and 'OUTAGE.START.TIME' into one column called 'OUTAGE.START'. And columns 'OUTAGE.RESTORATION.DATE' and 'OUTAGE.RESTORATION.TIME' into one column called 'OUTAGE.RESTORATION'.


#### Outages DataFrame

|   YEAR |   MONTH | U.S._STATE   | POSTAL.CODE   | NERC.REGION   | CLIMATE.REGION     |   ANOMALY.LEVEL | CLIMATE.CATEGORY   | OUTAGE.START.DATE   | OUTAGE.START.TIME   | OUTAGE.RESTORATION.DATE   | OUTAGE.RESTORATION.TIME   | CAUSE.CATEGORY     | CAUSE.CATEGORY.DETAIL   |   HURRICANE.NAMES |   OUTAGE.DURATION |   DEMAND.LOSS.MW |   CUSTOMERS.AFFECTED |   RES.PRICE |   COM.PRICE |   IND.PRICE |   TOTAL.PRICE |   RES.SALES |   COM.SALES |   IND.SALES |   TOTAL.SALES |   RES.PERCEN |   COM.PERCEN |   IND.PERCEN |   RES.CUSTOMERS |   COM.CUSTOMERS |   IND.CUSTOMERS |   TOTAL.CUSTOMERS |   RES.CUST.PCT |   COM.CUST.PCT |   IND.CUST.PCT |   PC.REALGSP.STATE |   PC.REALGSP.USA |   PC.REALGSP.REL |   PC.REALGSP.CHANGE |   UTIL.REALGSP |   TOTAL.REALGSP |   UTIL.CONTRI |   PI.UTIL.OFUSA |   POPULATION |   POPPCT_URBAN |   POPPCT_UC |   POPDEN_URBAN |   POPDEN_UC |   POPDEN_RURAL |   AREAPCT_URBAN |   AREAPCT_UC |   PCT_LAND |   PCT_WATER_TOT |   PCT_WATER_INLAND |
|-------:|--------:|:-------------|:--------------|:--------------|:-------------------|----------------:|:-------------------|:--------------------|:--------------------|:--------------------------|:--------------------------|:-------------------|:------------------------|------------------:|------------------:|-----------------:|---------------------:|------------:|------------:|------------:|--------------:|------------:|------------:|------------:|--------------:|-------------:|-------------:|-------------:|----------------:|----------------:|----------------:|------------------:|---------------:|---------------:|---------------:|-------------------:|-----------------:|-----------------:|--------------------:|---------------:|----------------:|--------------:|----------------:|-------------:|---------------:|------------:|---------------:|------------:|---------------:|----------------:|-------------:|-----------:|----------------:|-------------------:|
|   2011 |       7 | Minnesota    | MN            | MRO           | East North Central |            -0.3 | normal             | 2011-07-01 00:00:00 | 17:00:00            | 2011-07-03 00:00:00       | 20:00:00                  | severe weather     | nan                     |               nan |              3060 |              nan |                70000 |       11.6  |        9.18 |        6.81 |          9.28 | 2.33292e+06 | 2.11477e+06 | 2.11329e+06 |   6.56252e+06 |      35.5491 |      32.225  |      32.2024 |         2308736 |          276286 |           10673 |           2595696 |        88.9448 |        10.644  |       0.411181 |              51268 |            47586 |          1.07738 |                 1.6 |           4802 |          274182 |       1.75139 |             2.2 |      5348119 |          73.27 |       15.28 |           2279 |      1700.5 |           18.2 |            2.14 |          0.6 |    91.5927 |         8.40733 |            5.47874 |
|   2014 |       5 | Minnesota    | MN            | MRO           | East North Central |            -0.1 | normal             | 2014-05-11 00:00:00 | 18:38:00            | 2014-05-11 00:00:00       | 18:39:00                  | intentional attack | vandalism               |               nan |                 1 |              nan |                  nan |       12.12 |        9.71 |        6.49 |          9.28 | 1.58699e+06 | 1.80776e+06 | 1.88793e+06 |   5.28423e+06 |      30.0325 |      34.2104 |      35.7276 |         2345860 |          284978 |            9898 |           2640737 |        88.8335 |        10.7916 |       0.37482  |              53499 |            49091 |          1.08979 |                 1.9 |           5226 |          291955 |       1.79    |             2.2 |      5457125 |          73.27 |       15.28 |           2279 |      1700.5 |           18.2 |            2.14 |          0.6 |    91.5927 |         8.40733 |            5.47874 |
|   2010 |      10 | Minnesota    | MN            | MRO           | East North Central |            -1.5 | cold               | 2010-10-26 00:00:00 | 20:00:00            | 2010-10-28 00:00:00       | 22:00:00                  | severe weather     | heavy wind              |               nan |              3000 |              nan |                70000 |       10.87 |        8.19 |        6.07 |          8.15 | 1.46729e+06 | 1.80168e+06 | 1.9513e+06  |   5.22212e+06 |      28.0977 |      34.501  |      37.366  |         2300291 |          276463 |           10150 |           2586905 |        88.9206 |        10.687  |       0.392361 |              50447 |            47287 |          1.06683 |                 2.7 |           4571 |          267895 |       1.70627 |             2.1 |      5310903 |          73.27 |       15.28 |           2279 |      1700.5 |           18.2 |            2.14 |          0.6 |    91.5927 |         8.40733 |            5.47874 |
|   2012 |       6 | Minnesota    | MN            | MRO           | East North Central |            -0.1 | normal             | 2012-06-19 00:00:00 | 04:30:00            | 2012-06-20 00:00:00       | 23:00:00                  | severe weather     | thunderstorm            |               nan |              2550 |              nan |                68200 |       11.79 |        9.25 |        6.71 |          9.19 | 1.85152e+06 | 1.94117e+06 | 1.99303e+06 |   5.78706e+06 |      31.9941 |      33.5433 |      34.4393 |         2317336 |          278466 |           11010 |           2606813 |        88.8954 |        10.6822 |       0.422355 |              51598 |            48156 |          1.07148 |                 0.6 |           5364 |          277627 |       1.93209 |             2.2 |      5380443 |          73.27 |       15.28 |           2279 |      1700.5 |           18.2 |            2.14 |          0.6 |    91.5927 |         8.40733 |            5.47874 |
|   2015 |       7 | Minnesota    | MN            | MRO           | East North Central |             1.2 | warm               | 2015-07-18 00:00:00 | 02:00:00            | 2015-07-19 00:00:00       | 07:00:00                  | severe weather     | nan                     |               nan |              1740 |              250 |               250000 |       13.07 |       10.16 |        7.74 |         10.43 | 2.02888e+06 | 2.16161e+06 | 1.77794e+06 |   5.97034e+06 |      33.9826 |      36.2059 |      29.7795 |         2374674 |          289044 |            9812 |           2673531 |        88.8216 |        10.8113 |       0.367005 |              54431 |            49844 |          1.09203 |                 1.7 |           4873 |          292023 |       1.6687  |             2.2 |      5489594 |          73.27 |       15.28 |           2279 |      1700.5 |           18.2 |            2.14 |          0.6 |    91.5927 |         8.40733 |            5.47874 |

# <u>Baseline Model</u>


### Features
Since I will be building a Decision Tree Classifier, I will pick 2 features that I believe will be good variables to predict the category cause of future major outages. The features I want to use to train my model are 'MONTH' and 'NERC.REGION'. I chose these features because after performing a quick exploratory data analysis, I noticed a significant amount of the cause category of major outages was due to severe weather. So I chose two variables that I believe reflect my findings. Since these features give insight into the time of the year and specific locations that the outages occurred in would provide useful information to decide whether outages occurred due to severe weather or not. In my features, I have two nominal categorical features.

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

#### Feature Engineering & Preprocessing

My categorical features need to be transformed so that they can be used in my DTC model. Since all of my categorical features are nominal I will one hot encode them to transform them from a categorical feature into a numerical feature so that my model can use them.

#### Baseline Pipeline
I will be splitting my data into two sections training and testing data. Where 0.75% of my data will be dedicated to training and 0.25% will be dedicated to validation. My model will be trained on purely the training data but I will access the accuracy of the model using both training and testing data to compare the two results. To build my model I will be using a pipeline to preprocess the categorical features into numerical ones to train a decision tree classifier.

### Summary of Results

My current baseline model has an accuracy of 0.605 on training data and a score of 0.536 on testing data. This tells me my model does not predict the outcome of training data that well and on new never seen data my model performs even worse. This tells me the model is slightly overfitted with the training data and needs more features since it is only accurate 53.6% percent of the time when tested on new data. So I would not consider my baseline model as "good."

# <u>Final model</u>

### New Features

As previously mentioned to improve my baseline model I needed more features that give more insight into what the category cause of the major outage was. I did some more exploratory data analysis and found that the second and third-highest counts of category causes of major outages were intentional attacks and system operability disruption. These category causes are more general than severe weather so I looked for more features that could reflect that. The new features I chose were, 'CLIMATE.REGION', 'POPULATION', 'HURRICANE.NAMES', 'AREAPCT_URBAN', and 'PCT_WATER_TOT'. For the features 'U.S._STATE', 'POPULATION', and 'AREAPCT_URBAN' since these features reflect the demographics of the people affected by the major outages I believe it would better explain the high number of major outages due to intentional attacks and system operability disruption. As for the features 'CLIMATE.REGION', 'HURRICANE.NAMES', and 'PCT_WATER_TOT' since severe weather is still the most common reason why major outages occurred I believe this would provide my model better information on the weather and land conditions so that it predicts if major outages occurred due to severe weather or not more accurately.



### Preprocessing Line Additions
<ul>
  <li> Custom Function Transformer that binarizes the 'HURRICANE.NAMES' column.</li>
  <li> One hot encodes 'U.S._STATE' and 'CLIMATE.REGION' columns.</li>
  <li> Transforms the 'POPULATION' column into quantiles.</li>
  <li> Leaves 'AREAPCT_URBAN', and 'PCT_WATER_TOT' columns as is.</li>
</ul>




### Decision Tree Classifier Hyperparameter Fine Tuner

To further improve my model I decided I want to fine-tune the hyperparameters of the Decision Tree Classifier. Those being the max depth and the minimum sample split. The reason why I chose to tune the max depth is that I want to make my tree more expressive since there are multiple cause categories I want my tree to be a bit more complex. Secondly, the reason why I chose to fine-tune the minimum sample split of the tree is that it is not guaranteed that my tree will be built symmetrically. As a result, I want to optimize generalization performance by increasing the number of minimum sample splits. This allows some branches to grow deeper than others producing more tree splits to better classify the samples.
## Summary of Final Model and Results

### Final Model Breakdown

In conclusion, the final model I chose was a Decision tree Classifier. The features I chose for my model were 'MONTH', 'U.S._STATE', 'NERC.REGION', 'CLIMATE.REGION', 'POPULATION', 'HURRICANE.NAMES', 'AREAPCT_URBAN', and 'PCT_WATER_TOT'. Five of those features were nominal categorical features, while three of them were numerical features. I one hot-encoded four of the nominal categorical features to turn them into numerical ones. I binarized 'HURRICANE.NAMES' using a custom function transformer I made to binarize its values. As for the numerical features, I turned the 'POPULATION' feature into quantiles and left 'AREAPCT_URBAN', and 'PCT_WATER_TOT' as is. As for the hyperparameters of the Decision Tree Classifier, I ended up choosing a max depth of 10 and a minimum sample split of 5. The way I did this was by performing a grid search, the grid search took my unfit pipeline and a dictionary of ranges of values for the hyperparameters I wanted to fine-tune. Then it performed a k-fold cross-validation to find the combination of hyperparameters with the best average validation performance.
### Results Breakdown

As for the results, I saw significant improvement in both the training data and testing data accuracy. The accuracy of the training data was 0.804 and for the testing data, it was 0.589. This tells me that the new features I included and engineered alongside the hyperparameters I fine-tuned were able to better optimize generalization performance. My model became more generalized allowing for better predictions of the category cause of major outages on both the training and unseen data. My last model was too specific as it only had access to two features that relate mostly to the severe weather causes but not the other causes. But my final model was able to become more generalized as it had more features to make better predictions since they gave more insight into both the weather conditions and also demographics of the population affected by the outage.

# <u> Fairness Analysis</u>

To access the fairness of my final model, I want to see whether my model is fair when predicting the cause category of major outages between low and high-population states. I will continue to use accuracy as my evaluation metric to conduct my fairness analysis. Since there is no exact definition of low and high-population states I define my definitions here. I first created a new column with the quantiles of the population for each row. I then defined Low-population states as states with a Population quantile of three or lower, and High Population states as states with Population Quantiles greater than 3. I will be using the absolute difference in accuracy as my test statistic. Additionally, I will choose a significance level of 0.05 as a cut-off for my p-value since a p-value smaller than 0.05 indicates strong evidence against my null hypothesis. Lastly, to conduct my fairness analysis I will use a permutation test to test my hypotheses.


### Hypotheses:

<b>Null Hypothesis: </b>The classifier's accuracy is the same for both low population states and high population states, and any differences are due to chance.

<b>Alternative Hypothesis: </b>There is a difference in accuracy for low population states and high population states.

<b>Observed Absolute Difference in Accuracy: </b>0.0411

### Set Up
To begin my fairness analysis, I first need to create a new column that contains the Population Quantiles of the state that the major outages occurred in. Once I had that, I then turned the column into a Boolean column, on the condition if a row had a value of 3 or lower it will be cast as True else False. Now that I have my two groups, high and low population states I was able to begin my analysis.

### Summary of Results

The plot below shows the results of my permutation test. It displays the empirical distribution of the generated absolute differences in accuracy under the null. The red line shows the observed value. The p-value I calculated was 0.46.

<iframe src="assets/emperical.html" width=800 height=600 frameBorder=0></iframe>

Since the p-value is greater than the significance level, 0.46 > 0.05, we fail to reject the null hypothesis. There is not enough evidence to suggest that there may be a difference in accuracy between low and high-population states.
