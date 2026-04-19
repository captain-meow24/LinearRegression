/*
step 1: make a loss function that will be MSE
Error = (summation of(y - y')^2)/m   [m is the total number of rows]
step 2: the linear regression function will take in x_train vector<vector<int>> and y_train (x being the data and y being the known target)
We will guess weights and bias firstly as 0 and make a prediction based on that
We differentiate the MSE and find the rate of change of error with respect to weight and bias
We calculate the new weights and bias such that-
   w = w - learning_rate * differentiation (with respect to the particular weight)
   we subtract so that if error varries negatively with weight/bias -> weight should be increased and vice versa
   iterate through the each row and then iterate through the whole table as many times as there are number of epochs
*/

#include<iostream>
#include <vector>
using namespace std;

class LinearRegressor {
public:
   vector<double> weights;
   double bias = 0.0;
   double learning_rate = 0.0;
   int epochs = 0;
   LinearRegressor(int epochs = 1000, double learning_rate = 0.01);
   void gradient_descent(vector<vector<double>>& x_train, vector<double>& y_train);
};