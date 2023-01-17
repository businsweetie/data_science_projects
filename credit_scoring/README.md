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
[]()  
[]()  
[]()  

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

<a name="сonclusion"><h2>Conclusion</h2></a>
After reviewing the methods for selecting features, it was found that both methods select approximately the same features.
