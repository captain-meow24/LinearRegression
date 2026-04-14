//
// Created by kanishka on 14/4/26.
//

#ifndef LINEARREGRESSOR_H
#define LINEARREGRESSOR_H

#include<iostream>
#include<vector>
using namespace std;

class LinearRegression {
    vector<double> input_features;  //these are the features we are predicting for
public:
    LinearRegression(vector<vector<double>>& input, vector<double>& actual_val);
    double find_error(double pred, double actual);
    pair<vector<double>,double> gradient_descent(double learn_rate, int epochs, vector<vector<double>>& input, vector<double>& act_v, vector<double>& w, double b);
    double predicted_val = 0;
    void get_features(int n);
};

#endif //LINEARREGRESSOR_H
