//
// Created by kanishka on 14/4/26.
//

#include "LinearRegressor.h"

double LinearRegression::MSE(double pred, double actual) {
    double err = actual - pred;
    return err * err;
}

LinearRegression::LinearRegression(vector<vector<double> > input, vector<double> actual_val) {
    double learn_rate = 0.0001;   //we are keeping learning rate so small to account for differentiate values and to prevent overshooting
    int epochs = 1000;   //no. of times we iterate over a data set
    int n = input[0].size();
    vector<double> w(n,0.0);
    double bias  = 0.0;
    pair<vector<double>, double> result;
    result = gradient_descent(learn_rate, epochs, input, actual_val, w, bias);
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

pair<vector<double>, double> LinearRegression::gradient_descent(double learn_rate, int epochs, vector<vector<double>> input, vector<double> act_v, vector<double> w, double b) {
    int row = input.size();
    int feat = input[0].size();
    for (int i=0;i<epochs;i++) {
        for (int j=0; j<row; j++) {
            double predic =0;
            for (int k=0;k<feat;k++) {
                predic += w[k]* input[j][k];
            }
            predic += b;
            double error = MSE(predic, act_v[j]);
            //differentiating the error
            for (int k=0; k<n; k++){
                w[k] += -(1.0/row)*(x[j][k])*error;
            }
            b += -(1.0/row)*error;
        }
    }
}



