### Description

This is a team project. You are encouraged to form teams in any way you like, but each team must consist of either 4 or 5 people.

# Project Summary

Online hate speech is an important issue that breaks the cohesiveness of online social communities and even raises public safety concerns in our societies. Motivated by this rising issue, researchers have developed many traditional machine learning and deep learning methods to detect hate speech on online social platforms automatically.

Essentially, the detection of online hate speech can be formulated as a text classification task: "_Given a social media post, classify if the post is hateful or non-hateful_". In this project, you are to apply machine learning approaches to perform hate speech classification. Specifically, you will need to perform the following tasks.

# Task 1: Implement Logistics Regression (10 marks)

Recalled that you have learned about Logistic Regression in your earlier class. Your task is to implement a Logistic Regression model **from scratch**. Note that you are **NOT TO USE** the sklearn logistic regression package or any other pre-defined logistic regression package for this task! Usage of any logistic regression packages will result in 0 marks for this task.

### Key Task Deliverables

1a. Code implementation of the Logistic Regression model.  
1b. Prediction made by your Logistic Regression on the Test set. Note that you are welcome to submit your predicted labels to Kaggle but you will need to submit the final prediction output in the final project submission. Please label the file as "_LogRed_Prediction.csv_".

### Tips

- Check out the Logistic Regression implementation in this [awesome blog](https://towardsdatascience.com/logistic-regression-from-scratch-in-python-ec66603592e2).
- Your implementation should have the following functions:  
    -- `sigmoid(z)`: A function that takes in a Real Number input and returns an output value between 0 and 1.  
    -- `loss(y, y_hat)`: A loss function that allows us to minimize and determine the optimal parameters. The function takes in the actual labels _y_ and the predicted labels _y_hat_, and returns the overall training loss. Note that you should be using the Log Loss function taught in class.  
    -- `gradients(X, y, y_hat)`: The Gradient Descent Algorithm to find the optimal values of our parameters. The function takes in the training feature _X_, actual labels _y_ and the predicted labels _y_hat_, and returns the partial derivative of the Loss function with respect to weights (_w_) and bias (_db_).  
    -- `train(X, y, bs, epochs, lr)`: The training function for your model.  
    -- `predict(X)`: The prediction function where you can apply your validation and test sets.

# Task 2: Apply dimension reduction techniques (10 marks)

Dimension reduction is the transformation of data from a high-dimensional space into a low-dimensional space so that the low-dimensional representation retains some meaningful properties of the original data. The _train_ dataset contains 5000 TD-IDF features. In this task, you are to apply PCA to reduce the dimension of features.

### Key Task Deliverables

2a. Code implementation of PCA on the _train_ and _test_ sets. Note that you are allowed to use the [sklearn package](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) for this task.  
2b. Report the Macro F1 scores for applying 2000, 1000, 500, and 100 components on the test set. Note that you will have to submit your predicted labels to Kaggle to retrieve the Macro F1 scores for the test set and report the results in your final report. Use KNN as the machine learning model for your training and prediction (You are also allowed to use the [sklearn package for KNN implementation](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)) (set n_neighbors=2).

# Task 3: Try other machine learning models and race to the top! (25 marks)

In this course, you are exposed to many other machine learning models. For this task, you can apply any other machine learning models (taught in the course or not) to improve the hate speech classification performance! Nevertheless, you are **NOT TO** use any deep learning approach (if you are keen on deep learning, please sign up for the Deep Learning course! - highly encourage!).

To make this task fun, we will have a race to the top! Bonus marks will be awarded as follows:

- 1 mark: For the third-highest score on the private leaderboard.
- 2 marks: For the second-highest score on the private leaderboard.
- 3 marks: For the top-highest score on the private leaderboard.

Note that the private leaderboard will only be released after the project submission. The top 3 teams will present their solution on week 13 to get the bonus marks!

### Key Task Deliverables

3a. Code implementation of all the models that you have tried. Please include comments on your implementation (i.e., tell us the models you have used and list the key hyperparameter settings.  
3b. Submit your predicted labels for the test set to Kaggle. You will be able to see your model performance on the public leaderboard. Please make your submission under your registered team name! We will award the points according to the ranking of the registered team name.

# Task 4: Documenting your journey and thoughts (5 marks)

All good projects must come to an end. You will need to document your machine learning journey in your final report. Specifically, please include the following in your report:

1. An introduction of your best performing model (how it works)
2. How did you "tune" the model. Discuss the parameters that you have used and the different parameters that you have tried before arriving at the best results.
3. Did you self-learned anything that is beyond the course? If yes, what are they, and do you think if it should be taught in future Machine Learning courses.

### Key Task Deliverables

4a. A final report (in PDF) answering the above questions.

## Acknowledgements

This dataset is from an awesome group of researchers from UC San Diego and Georgia Institute of Technology.

ElSherief, M., Ziems, C., Muchlinski, D., Anupindi, V., Seybolt, J., De Choudhury, M., & Yang, D. (2021, November). Latent Hatred: A Benchmark for Understanding Implicit Hate Speech. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (pp. 345-363).

### Evaluation

Predictions are evaluated by the Macro F1 score between predicted class labels (i.e., hateful or non-hateful) and actual class labels in the test set. Specifically, the F1 score for each class is computed as follows:

Where TP=True Positive, FP=False Positive, and FN=False Negative. Next we compute the macro average of the *Hateful F1 Score* and *Non-Hateful F1 Score*. You may refer to this [blog](https://towardsdatascience.com/micro-macro-weighted-averages-of-f1-score-clearly-explained-b603420b292f) for more detail explanation of the evaluation metric. ### Submission File For each id in the test set (representing a post), you must predict its class, where 1=hateful and 0=non-hateful. The file should contain a header and have the following format: > id, label > 17185,0 > 17186,1 > 17187,0 > 17188,1 > 17189,1 > ...

### Deliverables And Deadlines

### Final Deliverables

Each team must submit the following:

1. Jupyter Notebook file with your code implementation for Task 1-3. Please segment and label the codes accordingly.
2. Final prediction output (csv file) that your team has submitted to Kaggle for Task 3.
3. Project report detailing the required information mentioned in Task 4.

### Deadline

**02 Aug 2024 (11:59PM):** Submission of Jupyter Notebook file + Final prediction output  
**11 Aug 2024 (11:59PM):** Submission of project report

### Grading Metric




| Task 1                                                                                                                       |                                                                                            |                                                                                                                                                                                |                                                                                                                                                                                                                               |
| ---------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0 Marks: No submission or utilized SKlearn logistic regression package or any other pre-defined logistic regression package. | 1-4 Marks : Incorrectly implemented Logistic Regression algorithm with compilation errors. | 5-7 Marks : Implemented the Logistic Regression algorithm with minor errors. Implemented model is compilable but has issues training the implemented model with the train set. | 8-10 Marks : Perfect Implementation of the Logistics Regression algorithm. Successfully trained the implemented model with the train set and achieved comparative performance compared to SKLearn Logistic Regression package |

| Task 2                   |                                                                                                                                                        |                                                                                                                                             |                                                                                                                                                                                    |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0 Marks : No submission. | 1-4 Marks : Incorrectly implement PCA and KNN. Unable to reduce the train set feature dimensions and run KNN on the test sets with reduced components. | 5-7 Marks : Correctly implemented the PCA and KNN with minor errors. Implemented model is able to run on test sets with reduced components. | 8-10 Marks : Perfectly implemented the PCA and KNN. Implemented model is able to run on test sets with reduced components and performed detail analysis of the reduced components. |



| Task 3*                  |                                                                                                                                                     |                                                                                                                                                                                                            |                                                                                                                                                                                                                         |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0 Marks : No submission. | 1-12 Marks : The final submission is not able to beat Red baselines in the private leaderboard. The team have explored 1-2 machine learning models. | 13-20 Marks : The final submission is able to beat Red baseline in the private leaderboard. The team have explored 2-3 solutions machine learning models with proper documentation on the hyperparameters. | 20-25 Marks : The final submission is able to beat Red and Blue baseline in the private leaderboard. The team have explored more than 4 machine learning models with proper documentation on the hyperparameter tuning. |

*For Task 3, you will be able to get bonus marks (cap to 25 marks)

- 1 mark: For the third-highest score on the private leaderboard.
- 2 marks: For the second-highest score on the private leaderboard.
- 3 marks: For the top-highest score on the private leaderboard.



| Task 4                   |                                                                                       |                                                                                                                                                           |                                                                                                                                                                                              |
| ------------------------ | ------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0 Marks : No submission. | 1-2 Marks : The documentation is inadequate to explain the models explored in task 3. | 3-4 Marks : The documentation is adequate to explain the models explored in task 3. The documentation also answered the questions listed under this task. | 5 Marks : The documentation is adequate to explain the models explored in task 3. The documentation also answered the questions listed under this task creatively with interesting insights. |
