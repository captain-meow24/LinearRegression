//
// Created by kanishka on 14/4/26.
//

#include "LinearRegressor.h"

double LinearRegression::MSE(vector<double> pred, vector<double> actual) {
    int num = pred.size();
    double error;
    for (int i=0;i<num;i++) {
        int err = actual[i] - pred[i];
        err *= err;
        error +=err;
    }
    return error;
}

LinearRegression::LinearRegression(vector<vector<double> > input, vector<double> actual_val) {
    double learn_rate = 0.0001;   //we are keeping learning rate so small to account for differentiate values and to prevent overshooting
    int epochs = 1000;   //no. of times we iterate over a data set
    int n = input[0].size();
    vector<double> w(n,0.0);
    double bias  = 0.0;
    pair<vector<double>, double> result;
    result = gradient_descent(learn_rate, epochs, input, actual_val);
    get_features();
    for (int i=0;i<n;i++) {
        predicted_val += result.first[i] * input_features[i];
    }
    predicted_val += bias;
}
