# Housing-price-prediction

INTRODUCTION:
Purchasing a home remains one of the biggest and most valuable purchasing decisions that individuals make in their lifetime.
Our project serves the purpose of house pricing analysis and prediction using various regression algorithms, alongside with analytical comparison of the implemented regression models.
The insights gathered could be used by individuals to complement their decision making process when purchasing a house. This helps to maximise the value users can gain while keeping to a budget. Findings could also be used by a property agent to improve the probability of a sale through proper marketing of key variables. 
We are making the use of ‘Boston housing dataset’ (available on kaggle) which consists of mainly categorical variables stored as integers and factors. We would be focusing on descriptive and predictive analytics, with suggestions on how additional data could be obtained to conduct more experiments to optimise the purchase. 

AIM OF THE PROJECT: 
The aim of our project is the Prediction of House Prices using Advanced Regression Techniques. Making the use of Data Visualisation, Feature Engineering, Linear Regression, Random forest regression and Gradient boosting regression.

The project has 3 parts-
1. Data prepeartion and feature engineering
2. Implementing pregression models
3. comparission of regression models 

ALGORITHM USED: 
MULTIPLE LINEAR REGRESSION:
Linear regression may be defined as the statistical model that analyzes the linear relationship between a dependent variable with a given set of independent variables. Linear relationship between variables means that when the value of one or more independent variables will change (increase or decrease), the value of dependent variable will also change accordingly (increase or decrease).
Mathematically the relationship can be represented with the help of following equation −
Y = mX + b
Consider a dataset having n observations, p features i.e. independent variables and y as one response i.e. dependent variable the regression line for p features can be calculated as follows −
 h(xi)=b0+b1xi1+b2xi2+...+bpxip
Here, h(xi) is the predicted response value and b0,b1,b2…,bp are the regression coefficients

RANDOM FOREST:
Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes or mean/average prediction of the individual trees.
It uses bagging and feature randomness when building each individual tree to try to create an uncorrelated forest of trees whose prediction by committee is more accurate than that of any individual tree. 
The random forest is a model made up of many decision trees. Rather than just simply averaging the prediction of trees (which we could call a “forest”), this model uses two key concepts that gives it the name random: Random sampling of training data points when building trees.
GRADIENT BOOSTING:
Gradient boosting is a machine learning technique for regression and classification problems, which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees. It relies on the intuition that the best possible next model, when combined with previous models, minimizes the overall prediction error. 
The key idea is to set the target outcomes for this next model in order to minimize the error. Gradient boosting is a greedy algorithm and can overfit a training dataset quickly. It can benefit from regularization methods that penalize various parts of the algorithm and generally improve the performance of the algorithm by reducing overfitting.

ADVANTAGE/DISADVANTAGE OF THE ALGORITHM:
ADVANTAGES OF RANDOM FOREST
Random Forest is based on the bagging algorithm and uses Ensemble Learning technique. It creates as many trees on the subset of the data and combines the output of all the trees. In this way it reduces overfitting problems in decision trees and also reduces the variance and therefore improves the accuracy.
Random Forest can be used to solve both classification as well as regression problems, i.e. It works well with both categorical and continuous variables.
Random Forest can automatically handle missing values. 
DISADVANTAGES OF RANDOM FOREST-
Complexity: It creates a lot of trees (unlike only one tree in case of decision trees) and combines their outputs. By default, it creates 100 trees in the Python sklearn library. To do so, this algorithm requires much more computational power and resources.
Longer Training Period: Requires much more time to train as compared to decision trees as it generates a lot of trees (instead of one tree in case of decision tree) and makes decisions on the majority of votes.

ADVANTAGES OF GRADIENT BOOSTING
Trees are built one at a time, where each new tree helps to correct errors made by previously trained trees. With each tree added, the model becomes even more expressive. There are typically three parameters - number of trees, depth of trees and learning rate, and each tree built is generally shallow.
DISADVANTAGES OF GRADIENT BOOSTING
Gradient Boosting Models will continue improving to minimize all errors. This can overemphasize outliers and cause overfitting.
Computationally expensive - often require many trees (>1000) which can be time and memory exhaustive. Also, training generally takes longer because of the fact that trees are built sequentially.
 
1. DATA ANALYSIS
	Dataset: Boston house prices dataset
	Source: sklearn Library

Boston house prices dataset
---------------------------

**Data Set Characteristics:**  

    :Number of Instances: 506 

    :Number of Attributes: 13 numeric/categorical predictive. Median Value (attribute 14) is usually the target.

    :Attribute Information (in order):
        - CRIM     per capita crime rate by town
        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
        - INDUS    proportion of non-retail business acres per town
        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
        - NOX      nitric oxides concentration (parts per 10 million)
        - RM       average number of rooms per dwelling
        - AGE      proportion of owner-occupied units built prior to 1940
        - DIS      weighted distances to five Boston employment centres
        - RAD      index of accessibility to radial highways
        - TAX      full-value property-tax rate per $10,000
        - PTRATIO  pupil-teacher ratio by town
        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
        - LSTAT    % lower status of the population
        - MEDV     Median value of owner-occupied homes in $1000's  
The dataset is loaded and converted into a dataframe 
Analysis and Feature Engineering is carried out on the data:
Checking for missing data, plotting the median values to check for side skewed data, plotting a correlation matrix to analyse feature dependencies followed by normalisation



CONCLUSION
With the various predictive models, it could be possible that each model, while effective on its own, could only predict or capture certain aspects of the data. Furthermore, we were interested if ensembling various models together could allow us to obtain better predictions. Thus we decided to incorporate a simple weighted average ensemble model to reduce the errors. The models we have chosen are The Linear Regression model, The Gradient Boost and The Random Forest model. In this case we gave a higher weightage to the Random Forest model as it showed  a better RMSE score thus promising higher accuracy  as compared to the other models. But at the same time random forest model proved to be indeed of higher complexity and thus slower. Gradient boosting model showed faster results with not much compromise with accuracy. Linear regression was evidently the least successful regression model out of the tree with less accuracy and highest run time.
