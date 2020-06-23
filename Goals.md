# Goals

We want to create a model that predicts outcomes based on different types of ranking (ATP ratings, Elo ratings, ATP rank points).

This is a classification problem (0 for loss, 1 for win), and our model will learn from data.

This is a supervised classification problem.

We know that there are multiple algorithms but since we want to interpret the results and don't want a "black-box" model, we will choose between :
- linear regression
- logistic regression

Multiple research works treated about logistic regression for tennis outcomes, so we can have an idea on how to process :
- Clarke and Dyte, 2000, Using official ratings to simulate major tournaments
    - use a logistic regression on one variable (rank)
- Sipko, 2015, Machine Learning for the prediction of professional tennis matches
    - use a logistic regression on multiple variables (more advanced model)
    
We will first dive into the simpler model and see the results.