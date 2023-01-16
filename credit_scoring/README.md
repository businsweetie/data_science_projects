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
[]()  
[]()  
[]()  
[]()  
[]()  
[]()  

<a name="primary_data_analysis"><h2>Primary data analysis</h2></a>
After the initial data analysis, the date of application was removed, duplicates were removed. Omissions were found in the attribute containing information about the level of education and their replacement was made with the most frequently repeated value.

For the **"number of rejected applications"** attribute, an information content check was carried out. A feature is considered uninformative if there are most rows with the same values in the feature, which means that it does not carry useful information for the project. The selected feature turned out to be 82% uninformative, therefore it can be excluded.

Based on the initial analysis, outliers were detected in some features (measurements that stand out from the general sample), which were subsequently removed.

<a name="dependencies"><h2>Identification of dependencies between features</h2></a>
To better determine which features affect the probability of a client's default, it is necessary to consider the relationship between the target feature and the other features. Based on the graphical representation of dependencies, the features that have a greater impact on the value of the target feature were identified
