Q6: feature engineering, Customer Churn Prediction
Google Colab: paste solution.py into ONE cell, run it, pick the file customer_churn.csv when asked.
PC: pip install pandas numpy matplotlib openpyxl ; python solution.py
ASSUMPTION: your zip had NO customer churn file, so customer_churn.csv here is sample data I generated with assumed columns (Customer_ID, Tenure_Months, Monthly_Charges, Total_Charges, Contract_Type, Support_Calls, Churn). When you get the real file, change the file name, the column names and the ID column. The steps stay the same.
Target: Churn. (1 = customer left, 0 = stayed)
Thresholds (for example Age > 60) are my choices from the data ranges. Explain them in your viva.
