//
// Created by kanishka on 14/4/26.
//

#include "LinearRegressor.h"

double LinearRegression::MSE(vector<double> pred, vector<double> actual) {
    int num = pred.size();
    double error =0;
    for (int i=0;i<num;i++) {
        double err = actual[i] - pred[i];
        err *= err;
        error +=err;
    }
    return error/(2.0 * num);
}

LinearRegression::LinearRegression(vector<vector<double> > input, vector<double> actual_val) {
    double learn_rate = 0.0001;   //we are keeping learning rate so small to account for differentiate values and to prevent overshooting
    int epochs = 1000;   //no. of times we iterate over a data set
    int n = input[0].size();
    vector<double> w(n,0.0);
    double bias  = 0.0;
    pair<vector<double>, double> result;
    result = gradient_descent(learn_rate, epochs, input, actual_val);
    get_features(n);
    for (int i=0;i<n;i++) {
        predicted_val += result.first[i] * input_features[i];
    }
    predicted_val += result.second;
}

void LinearRegression::get_features(int n) {
    cout<< "Enter values for predicton "<<endl;
    for (int i=0;i<n;i++) {
        double val;
        cin>>val;
        input_features.push_back(val);
    }
}

