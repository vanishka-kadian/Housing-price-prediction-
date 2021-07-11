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

CONCLUSION
With the various predictive models, it could be possible that each model, while effective on its own, could only predict or capture certain aspects of the data. Furthermore, we were interested if ensembling various models together could allow us to obtain better predictions. Thus we decided to incorporate a simple weighted average ensemble model to reduce the errors. The models we have chosen are The Linear Regression model, The Gradient Boost and The Random Forest model. In this case we gave a higher weightage to the Random Forest model as it showed  a better RMSE score thus promising higher accuracy  as compared to the other models. But at the same time random forest model proved to be indeed of higher complexity and thus slower. Gradient boosting model showed faster results with not much compromise with accuracy. Linear regression was evidently the least successful regression model out of the tree with less accuracy and highest run time.
