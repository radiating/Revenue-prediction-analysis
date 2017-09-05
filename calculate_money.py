#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 10:54:22 2017

@author: tingzhu
"""

import pandas as pd
import numpy as np


# calculate loan for book purchase for each month
def calc_loan(original_purchase_order):
    quantity = np.array(original_purchase_order['quantity_purchased'])
    unit_cost = np.array(original_purchase_order['cost_to_buy'])   
    return sum(quantity*unit_cost)

# calculate mailing cost
def calc_mailCost(num_products):
    mail_unitCost = 0.6
    return num_products*mail_unitCost

# create a book_id and price look-up dictionary
def make_price_dict(purchase_order):
    return dict(zip(purchase_order['product_id'], purchase_order['retail_value']))

# calculate profit for last month
def calc_profit(assortment,purchase_order):
    price_dict=make_price_dict(purchase_order)
    product_id = assortment[assortment['purchased'] == True]['product_id']
    list_margin=[]
    for product in product_id:
        list_margin.append(price_dict[product]) 
    return sum(list_margin)
    
# calculate profit for next month
# here the "purchase_result" is predicted
def calc_profit_nextmonth(next_assortment,purchase_result,purchase_order,next_purchase_order):
    
    table=pd.concat([next_assortment,purchase_result],axis=1)
    
    margin_record = make_price_dict(purchase_order)
    margin_record2 = make_price_dict(next_purchase_order)
    
    # combine dictionaries in case there are new books or price changes
    z = margin_record.copy()
    z.update(margin_record2)

    product_id = table[table['purchased'] == True]['product_id']
    list_margin=[]
    for product in product_id:
        list_margin.append(z[product]) 
    return sum(list_margin)