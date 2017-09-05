#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 11:04:36 2017

@author: tingzhu
"""

import pandas as pd
import calculate_money as cm
import generate_features as gf
import build_clf as cf
import sys

    


def are_we_going_to_survive():
    
    original_purchase_order=pd.read_csv('original_purchase_order.csv')
    next_purchase_order=pd.read_csv('next_purchase_order.csv')
    customer_features=pd.read_csv('customer_features.csv')
    product_features=pd.read_csv('product_features.csv')
    last_month_assortment=pd.read_csv('last_month_assortment.csv')
    next_month_assortment=pd.read_csv('next_month_assortment.csv')
 
    # calculate costs from last month
    lastmonth_Loan=cm.calc_loan(original_purchase_order)
    lastmonth_MailCost_outgoing=cm.calc_mailCost(len(last_month_assortment))
    
    didnotbuy=last_month_assortment['purchased'].value_counts()[0]
    lastmonth_MailCost_incoming=cm.calc_mailCost(didnotbuy)

    # calculate profit from last month
    lastmonth_Profit=cm.calc_profit(last_month_assortment,original_purchase_order)
    
    # calculate potential costs for next month
    nextmonth_Loan=cm.calc_loan(next_purchase_order)
    nextmonth_MailCost_outgoing=cm.calc_mailCost(len(next_month_assortment))
    
    # predict which product would sell next month
    purchased_result=predict_nextmonthSale(customer_features,product_features,last_month_assortment,next_month_assortment)
    
    # calculate potential profit for next month
    nextmonth_Profit=cm.calc_profit_nextmonth(next_month_assortment,purchased_result,original_purchase_order,next_purchase_order)
    
    # calculate potential mailing costs from returned products
    nextmonth_didnotbuy=purchased_result['purchased'].value_counts()[0]
    nextmonth_MailCost_incoming=cm.calc_mailCost(nextmonth_didnotbuy)    
    
    
    print("\n\n------- Balance Report ------\n")
    print("-------- Last month ---------\n")
    print("Loan:\t\t\t -%.2f"%lastmonth_Loan)
    print("Mail (outgoing):\t -%.2f"%lastmonth_MailCost_outgoing)
    print("Mail (incoming):\t -%.2f"%lastmonth_MailCost_incoming)
    print("Profit:\t\t\t +%.2f"%lastmonth_Profit)
    print("-------- Next month ---------\n")
    print("Loan:\t\t\t -%.2f"%nextmonth_Loan)
    print("Mail (outgoing):\t -%.2f"%nextmonth_MailCost_outgoing)
    print("Mail (incoming):\t -%.2f"%nextmonth_MailCost_incoming)
    print("Profit:\t\t\t +%.2f"%nextmonth_Profit)
    
    
    # calculate revenue 
    balance = lastmonth_Profit + nextmonth_Profit \
              - (lastmonth_MailCost_outgoing + lastmonth_MailCost_incoming) \
              - (nextmonth_MailCost_outgoing + nextmonth_MailCost_incoming) \
              - (lastmonth_Loan+nextmonth_Loan)
    
    if balance > 0:
        print("\nBalance:\t\t\t+%.2f\n"%balance)
        print('Yes')
    else:
        print("\nBalance:\t\t\t-%.2f\n"%balance)
        print('No')
    

def predict_nextmonthSale(customer_features,product_features,last_month_assortment,next_month_assortment):
    
    # prepare data features using customer and product features
    # returns data features for both last month and next month
    features_type1_last, features_type2_last,features_type1_next, features_type2_next = gf.generate_features(customer_features,product_features, last_month_assortment,next_month_assortment)
    
    # build a random forest classifier using last month's data features
    rf_clf = cf.classify_randomForest(features_type2_last)
    
    # load data features of next month
    X=features_type2_next.iloc[:,2:]
    
    # predict purchase outcome and save as a dataframe
    purchased_result = pd.DataFrame(list(rf_clf.predict(X) ),columns=['purchased'])
    
    return purchased_result
    
if __name__ == "__main__":
    print("Run analysis")
    are_we_going_to_survive()
    
    
    
    
    
    
