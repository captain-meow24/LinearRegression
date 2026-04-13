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
