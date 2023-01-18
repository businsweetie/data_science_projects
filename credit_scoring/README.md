# Forecasting the solvency of bank customers using machine learning methods

In order to see the algorithms work in practice, we will use a data set from the site [kaggle.com about the clients of Tinkoff Bank](https://www.kaggle.com/c/fintech-credit-scoring), for which it is proposed to predict the fact of default according to the data from the questionnaire using machine learning algorithms.

The data set contains information about 205296 clients and 17 attributes, 5 quantitative, 7 categorical, 4 binary attributes and 1 predictive or dependent variable:
1. Age;
2. Scoring score: from -3.624586 to 0.199773; 
3. Number of requests to the bank; 
4. Client's income;
5. Number of rejected applications;
6. Number of connections with other clients of the same bank: 1, 2, 3, more than 3;
7. Level of education: SH – secondary school, SH – higher school, B – Bachelor's degree, M – Master's degree, A – postgraduate;
8. Client's gender: male, female;
9. The number of years that the applicant has been a client of the bank: 1, 2, 3, more than 3;
10. Type of client's living space: apartment, studio, house;
11. Current position: Director, Chief, specialist;
12. The region in which the client lives: Yakutia, Kamchatka, Kurgan, Orenburg, NSK – Novosibirsk, SPB – St. Petersburg, MSK - Moscow;
13. Availability of a car: yes, no;
14. Availability of a foreign car: yes, no;
15. Having a job with an above-average income: yes, no;
16. The presence of a foreign passport: yes, no;
17. Dependent variable the presence of default in the client: 0, 1.

## Content
[Primary data analysis](#primary_data_analysis)   
[Identification of dependencies between features](#dependencies)    
[Feature selection in the first way](#feature_selection_first)  
[Selection of features in the second way](#feature_selection_second)  
[Conclusion](#сonclusion)  
[Preparing data for model training](#preparing_data_for_train)  
[Metrics for evaluating the quality of the model](#metrics)  
[Training models](#train)  
[Undersampling method](#undersampling)

<a name="primary_data_analysis"><h2>Primary data analysis</h2></a>
After the initial data analysis, the date of application was removed, duplicates were removed. Omissions were found in the attribute containing information about the level of education and their replacement was made with the most frequently repeated value.

For the **"number of rejected applications"** attribute, an information content check was carried out. A feature is considered uninformative if there are most rows with the same values in the feature, which means that it does not carry useful information for the project. The selected feature turned out to be 82% uninformative, therefore it can be excluded.

Based on the initial analysis, outliers were detected in some features (measurements that stand out from the general sample), which were subsequently removed.

<a name="dependencies"><h2>Identification of dependencies between features</h2></a>
To better determine which features affect the probability of a client's default, it is necessary to consider the relationship between the target feature and the other features. Based on the graphical representation of dependencies, [the features that have a greater impact on the value of the target feature were identified](#dependencies_between_features_img).

<a name="dependencies_between_features_img">![DependenciesBetweenFeatures](https://github.com/businsweetie/data_science_projects/blob/main/credit_scoring/pic/dependencies_between_features.png)</a>

After the initial data analysis, it is necessary to carry out the selection of features, because not all features are needed for training algorithms, but only those that have a greater impact on the final result.

The following methods will be used to select features:
- [WoE (Weight of Evidence)](https://machinelearningmastery.ru/attribute-relevance-analysis-in-python-iv-and-woe-b5651443fc04/) with subsequent assessment of the predictive power of the selected factors using the IV (information value) algorithm;
- Evaluation of the importance of features using a random forest algorithm.

Before you start working on the selection of features in the first way, you need to convert all quantitative continuous features into categorical ones using quantization (binning). Quantization is a data processing process that allows you to split the range of a quantitative attribute into a given number of intervals (bins) and assign a name to each bin. There are two types of quantization:
- **Interval**. The range of values is divided into the same intervals, each will not have too much or too little data.
- **Quantile**. The width of the intervals will be different, but each will contain approximately the same number of values.

In this paper, the second type of quantization is used. Using quantization, the following attributes will be transformed: the age of the client, the scoring score and the client's income.

<a name="feature_selection_first"><h2>Feature selection in the first way</h2></a>

The WoE coefficient relative to this task characterizes the degree of deviation of the default level in this group from the average value in the sample.

To calculate WoE, for each categorical attribute and for each group within the attribute, calculate the number of customers with default ("bad" customers) and without default ("good" customers) and calculate the coefficient using the following formula:
$$WoE_i=\ln\left({\frac{p_i}{q_i}}\right),$$
where $i$ - the group number inside the attribute, $p$ - the share of "good" customers among all "good", $q$ - the share of "bad" customers among all "bad".

After calculating the WoE, the information value (coefficient IV) is calculated, which characterizes the statistical significance of the trait, according to the following formula:
$$IV=\displaystyle\sum_{i=1}^n (p_i-q_i)\ast WoE_i.$$

To determine the predictive power of a trait based on the calculation of IV, we use the following classification:
- $<0.02$ - absent,
- $0.02-0.1$ - low,
- $0.1-0.3$ - average,
- $>0.3$ - high.

For the features that were selected as the most significant, the [table](#IV_feat_table) with the values of the coefficient IV is presented below.
<a name="IV_feat_table"></a>
Feature                                            | Value of IV
:-------------------------------------------------:|:----------:
Number of connections with other clients           | $0.13$
Scoring score                                      | $0.27$
Type of living space                               | $0.11$
Type of position held                              | $0.10$
The number of years that the applicant is a client | $0.10$

After feature selection, dummy features are created using **dummy coding**. Let one of the features $x_j$ take $m$ values $\{b_1,\dots b_m\}$, then for each object $x^j$, you can replace the feature $x_i^j$ with $m-1$ features with values $\{0,1\}$:
$$Z_i^{b_k}=I\left[x_i^j=b_k\right] \text{, } k\in \{1,\dots, m-1\},$$
where $I\left[A\right]$ is the event indicator $A$.

The last step in the selection of features by the first method is the removal of strongly correlated features (of the two features, you need to leave those with a higher WoE coefficient value), for this purpose [correlation matrices](#matrix_of_corr_img) were built.

<a name="matrix_of_corr_img">![Matrix of correlations](https://github.com/businsweetie/data_science_projects/blob/main/credit_scoring/pic/matrix_of_corr.png)</a>

Thus, the final data set consists of 10 features selected in the first way: Living space: studio; Living space: house; Current position: chief;
Scoring: $[-2.387; -2.116]$; Scoring: $[-2.116; -1.865]$; Scoring: $[-1.865; -1.566]$; Scoring: more than $-1.566$; Connections with other clients: 2; Connections with other clients: 3; Connections with other clients: more than 3.

<a name="feature_selection_second"><h2>Selection of features in the second way</h2></a>

The second method evaluates the importance of each feature based on a random forest algorithm. To do this, you need to train the model on a training sample and calculate the out-of-bag error for each object of this sample. The error is averaged for each element over the entire random forest. The values of each attribute are mixed for all objects of the training sample and the error calculations are performed anew to assess the importance of the attribute. The more the accuracy of predictions decreases due to the exclusion or permutation of a feature, the more important this feature is.

$$FI^{(t)}(x_j)=\frac{\sum_{i \in OOB^{(t)}} I\left(y_i=\widehat{y}_i^{(t)}\right)}{|OOB^{(t)}|} - \frac{\sum_{i \in OOB^{(t)}} I\left(y_i=\widehat{y}_{i,\pi_j}^{(t)}\right)}{|OOB^{(t)}|},$$

where $OOB^{(t)}$ - out-of-bag error for the tree $t\in\{1,\dots, N\}$, $N$ - the number of trees in a random forest, $x_j$ - a sign for whose importance is evaluated, $\widehat{y}^{(t)}$ - prediction before deleting or rearranging a feature, $\widehat{y}_{i,\pi_j}^{(t)}$ - prediction after deleting or rearranging a feature.

Next, the importance of the feature is calculated for all trees in a random forest and can be presented in two forms: non-normalized and normalized:
$$FI(x_j)=\frac{1}{N}\displaystyle\sum_{t=1}^N FI^{(t)}(x_j) \text{,     } z_j = \frac{N\cdot FI(x_j)}{\sigma},$$
where $\sigma$ is the standard deviation of the differences.

The [table](#import_feat_table) provides calculations of coefficients for the five most important features. For further analysis, 24 signs were selected for which the normalized value is greater than 0.02.
<a name="import_feat_table"></a>
Feature                     | Non-Normalized Value | Normalized value
:--------------------------:|:--------------------:|:---------------:
Region: Moscow              | $227.8$              | $0.043968$
Living space: studio        | $207.4$              | $0.040031$
Gender: male                | $199.0$              | $0.038410$
Scoring: $>-1.566$          | $189.4$              | $0.036557$
Scoring: $[-1.865; -1.566]$ | $182.1$              | $0.035148$

The essence of the method: the model is trained on the initial set of features, evaluates their significance and excludes the least important feature, the process is repeated until the optimal or specified number of features is obtained, each feature is assigned a rank, the higher the rank, the more important the feature.

Thus, the final data set consists of 10 features selected in the second way: Education: High school; Living space: studio; Score: $[-2.116; -1.865]$; Score: $[-1.865; -1.566]$; Score: $> -1.566$; Communication with customers: more than 3; Availability of a car; Bank customer: more than 3; Region: St. Petersburg; Region: Moscow Time.

<a name="preparing_data_for_train"><h2>Preparing data for model training</h2></a>

To build models, you first need to [divide the sample](#sample_table) into a training sample, on which the model will be trained, and a test sample, on which we will check the quality of models, with respect to 80/20.

<a name="sample_table"></a>
Sample type | Number of "good" Customers | Number of "bad" Customers | Proportion of "bad" сustomers
:----------:|:--------------------------:|:-------------------------:|:----------------------------:
Source      | 144631                     | 18748                      | 11.48\%
Train       | 115652                     | 15051                      | 11.52\%       
Test        | 28979                      | 3697                       | 11.31\%

<a name="metrics"><h2>Metrics for evaluating the quality of the model</h2></a>

The operation of the model can be characterized using such quality criteria as: errors of the first and second kind, $accuracy$, $recall$, $precision$, $F_1-score$, $AUC$ $ROC$, $AUC$ $PR$, Gini index. Before considering metrics, we introduce an important concept of the [error matrix](#error_matrix_img).

<a name="error_matrix_img">![ErrorMatrix](https://github.com/businsweetie/data_science_projects/blob/main/credit_scoring/pic/error_matrix.png)</a>

The $accuracy$ metric is common to all classes and is not applicable in problems with an unbalanced sample, as in the problem under consideration

$$accuracy=\frac{TP+TN}{TP+FP+FN+TN}.$$

To correctly assess the quality of algorithms, you need to use the metrics $recall$, $precision$:

$$precision=\frac{TP}{TP+FP},\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ recall=\frac{TP}{TP+FN}.$$

$Recall$, shows what proportion of objects of a positive class the model predicted from all objects of a positive class. $Precision$ shows which proportion of objects that the model predicted as positive is really positive.

Also, when training the model, there are errors of the I-th and II-th kind $False\,Positive$ and $False\,Negative$. In the problem under consideration, an error of the I-th kind can be interpreted as a commercial risk associated with refusal to creditworthy customers. The type II error characterizes the credit risk associated with the number of non-creditworthy customers classified as creditworthy.
If $recall$ and $precision$ are equally significant for the problem, $F_1$ is used-a measure (the harmonic mean of two metrics $recall$ and $precision$):
$$F_1=\frac{2\ast precision\ast recall}{precision+recall}.$$

ROC curve is a graph showing the relationship between correctly classified objects of positive class $TPR$ and falsely positively classified objects of negative class $FPR$
$$TPR=\frac{TP}{TP+FN},\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ FP=\frac{FP}{FP+TN}.$$

The metric $ROC$ $AUC$ (Area Under Curve) measures the area under the [curve](#roc_curve_img) $ROC$, the greater the steepness of the $ROC$ curve, the larger the area under it and the better the model works.

<a name="roc_curve_img">![ROCAUC](https://github.com/businsweetie/data_science_projects/blob/main/credit_scoring/pic/ROC%20AUC.png)</a>

Based on the metric $ROC$ $AUC$, you can calculate another metric – the Gini index

$$Gini=2\ast\left(ROC\,AUC-0.5\right),$$

the higher the Gini index, the better the discriminating ability of the model.

$PR$ is a curve – a graph constructed in the coordinates $recall$ and $precision$. The area under the $PR$ [curve](#pr_curve_img) is better used for problems with an unbalanced sample.

<a name="pr_curve_img">![ROCPR](https://github.com/businsweetie/data_science_projects/blob/main/credit_scoring/pic/AUC%20PR.png)</a>

<a name="train"><h2>Training models</h2></a>

Let's build basic models for all algorithms. For the convenience of designations, we number the models. Model 1 - Logistic regression, Model 2 - Nearest Neighbor method, Model 3 - Random Forest, Model 4 - Support Vector Machine. The models were implemented using Python libraries (LogisticRegression, KNeighborsClassifier, RandomForestClassifier, SVC). The results of the work of the basic models show that the models do not differ much from each other in their predictive ability, therefore, for each model it is necessary to [select parameters]() that will improve them. Also, the balancing method was applied for models 1, 3 and 4. The regularization coefficient of the logistic regression turned out to be too large, the model could have been retrained, so it needs to be checked using cross-validation on 10 folds. After cross-validation, it turned out that the strong regularization coefficient had almost no effect on the predictive ability of the model, the values of the metrics changed quite a bit.

<a name="hyp_param_table"></a>
<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>Hyperparameter</th>
            <th>Value</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Model 1</td>
            <td>Regularization Coefficient</td>
            <td>$0.0064281$</td>
        </tr>
        <tr>
            <td>Model 2</td>
            <td>Number of Neighbors</td>
            <td>$267$</td>
        </tr>
        <tr>
            <td rowspan=4>Model 3</td>
            <td>Number of Trees</td>
            <td>$3 \times 10^{-5}$</td>
        </tr>
        <tr>
            <td>Number of features</td>
            <td>$log2$</td>
        </tr>
        <tr>
            <td>Tree depth</td>
            <td>$1200$</td>
        </tr>
        <tr>
            <td>Number of objects in leaves</td>
            <td>$4$</td>
        </tr>
      <tr>
            <td rowspan=3>Model 4</td>
            <td>Regularization Coefficient</td>
            <td>$1$</td>
        </tr>
        <tr>
            <td>Gamma</td>
            <td>$1$</td>
        </tr>
        <tr>
            <td>Kernel type</td>
            <td>$Gaussian$</td>
        </tr>
    </tbody>
</table>

The selected hyperparameters and the application of the balancing method to some algorithms significantly improved the [predictive ability](#metric_after_hyp_table_img) of the models.

<a name="metric_after_hyp_table_img"></a>
Metric      | Model 1    | Model 2    | Model 3    | Model 4
:----------:|:----------:|:----------:|:----------:|:--------:
precision   | $0.168052$ | $0.125000$ | $0.174900$ | $0.174773$
recall      | $0.667027$ | $0.000541$ | $0.602380$ | $0.602651$     
$F_1-score$ | $0.268467$ | $0.001077$ | $0.271089$ | $0.270964$
AUC PR      | $0.149768$ | $0.113148$ | $0.150343$ | -

<a>![ErrorMetrixAfterHypPar](https://github.com/businsweetie/data_science_projects/blob/main/credit_scoring/pic/error_matrix_after_hyp.png)</a>

<a name="undersampling"><h2>Undersampling method</h2></a>

Our sample is unbalanced: the number of "good" customers is 144631, the number of "bad" customers is 18748. Undersampling is a process in which a certain number of records are randomly removed from the majority class in order to match the number with the records of a smaller class.
After applying the method: the number of "good" customers is 18748, the number of "bad" customers is 18748.
[Below](#error_matrix_after_undersampling_img) are the error matrices for the models after applying the undersampling method on the first set of selected features and metrics for this models.

<a name="metric_after_undersampling_table_img"></a>
Metric      | Model 1    | Model 2    | Model 3    | Model 4
:----------:|:----------:|:----------:|:----------:|:--------:
precision   | $0.609229$ | $0.610981$ | $0.606604$ | $0.606906$
recall      | $0.666488$ | $0.561762$ | $0.690655$ | $0.689044$     
$F_1-score$ | $0.636573$ | $0.585339$ | $0.645907$ | $0.645372$
AUC PR      | $0.340000$ | $0.260000$ | $0.340000$ | -

<a>![ErrorMetrixAfterUndersampling](https://github.com/businsweetie/data_science_projects/blob/main/credit_scoring/pic/error_matrix_after_undersampling.png)</a>

Obviously, models on balanced data work much better, showing better metric values.

<a name="сonclusion"><h2>Conclusion</h2></a>
The application of a particular model depends on the specific task. In forecasting problems, nonlinear models such as random forest and the support vector machine show the best results. The algorithms considered in the paper have shown good predictive ability for use in forecasting the solvency of bank customers. In our work on the F1 metric, the random forest model showed itself best.

The other models also showed excellent results on both unbalanced and balanced data. Despite the advantages of the nearest neighbor method (simple implementation, good interpretation, hyperparameter tuning), it did not show very good results on the data compared to other methods, probably due to the problem of data imbalance. But on the data that was modified using the undersampling method and this model showed good results.

If we compare the methods of feature selection, then the first method, based on the calculation of the WoE and IV coefficients, is more informative and interpretable than the second (estimating the importance of each feature based on the random forest algorithm). There are no strong differences in the construction of models based on the features selected by the first and second methods, therefore, the further application of these methods may depend on the task at hand.
