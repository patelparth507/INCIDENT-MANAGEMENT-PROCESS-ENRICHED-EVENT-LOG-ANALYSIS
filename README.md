# INCIDENT MANAGEMENT PROCESS ENRICHED EVENT LOG ANALYSIS

Problem Statement:
Determine ways to Improve profits and productivity in the incident management system by reducing downtime.

Objective:
1. Predict days elapsed for a given incident
2. Identifying patterns in incidents

Data Pre-Processing:
1. Multiple records for each incident hence I have choosen to only include incident entries that are closed
2. Imputed missing values with the most appropriate value for the column's category, subcategory, assignment_group, and assigned_to (Imputed the missing values in those columns with the ones having state ”Resolved” and " sys_updated_at== resolved_at" of the same incident)
3. To improve model's performance I removed 0.06 quantile-based outliers from days elapsed column (days more than 110)

Feature Engineering:
1. Removed static text out of the entries ex:-  Caller XXXX, Opened by XXXX, Resolver XXXX, code 5.
2. Engineered variables with large number of categorical values using mean encoding technique in python programming.

Modeling:
1. Multiple Linear Regression
2. Ridge Regularization
3. LASSO Regularization
4. SVM
5. Decision Tree
6. Neural Network
