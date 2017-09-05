# revenue-prediction-analysis
Predict whether a (mock) monthly subscription company will survive or not.

The purpose of this analysis is predict whether a (mock) monthly book subscription company is going to be profitable or not. The company sends 5 books each month for free according to customer profile information about their book preferences. Customers pay for the books they keep, and send the rest back to the company. Since the company has to buy the books in advance, and pay for the shipping costs (both to the customers and for the returns), we're interestd in finding whether we can predict which books would be kept by the customers, and hence calculate whether profits would be greater than the costs (buying books and shipping) for the next month such that the company could continue to operate.

The analysis takes 3 steps:
1) feature engineering using customers and books information
2) build and validate machine learning models using hold-out test sets
3) predict outcome (buy or return) of books and thus calculate profits and costs

The analysis was written in Python 3. The books and customer information are not publically available. The scripts are provided and the analysis results are as follows:

/**********************************************************/

Run analysis

------ Random Forest Model Result ------

Search grid of parameters
{'max_depth': [None], 'max_features': [10, 25], 'min_samples_split': [15, 25], 'min_samples_leaf': [5, 20], 'bootstrap': [True], 'criterion': ['entropy']}

Best parameters set found on development set:

{'bootstrap': True, 'criterion': 'entropy', 'max_depth': None, 'max_features': 25, 'min_samples_leaf': 20, 'min_samples_split': 25}

                   precision    recall  f1-score   support

      False           0.80      0.85      0.82      4704
       True           0.69      0.61      0.65      2583
    avg / total       0.76      0.76      0.76      7287

------- Balance Report ------

-------- Last month ---------

Loan:                    -135546.42

Mail (outgoing):         -21600.00

Mail (incoming):         -13992.60

Profit:                  +151617.36


-------- Next month ---------

Loan:                    -24155.51

Mail (outgoing):         -33840.00

Mail (incoming):         -24837.60

Profit:                  +179645.09


Balance:                        +77290.32

/**********************************************************/


