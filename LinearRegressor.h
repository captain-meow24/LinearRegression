//
// Created by kanishka on 14/4/26.
//

#ifndef LINEARREGRESSOR_H
#define LINEARREGRESSOR_H

#include<iostream>
#include<vector>
using namespace std;

class LinearRegression {
public:
    LinearRegression(vector<vector<double>> input, vector<double> actual_val);
    double MSE(vector<double> pred, vector<double> actual);
    pair<vector<double>,double> gradient_descent(double learn_rate, int epoch, vector<vector<double>> input, vector<double> act_v);
};

#endif //LINEARREGRESSOR_H
