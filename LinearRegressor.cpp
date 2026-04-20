#include"LinearRegressor.h"

LinearRegressor::LinearRegressor(int epoch, double learning) {
    learning_rate = learning;
    epochs = epoch;
}

void LinearRegressor::gradient_descent(vector<vector<double>>& x_train, vector<double>& y_train) {
    int col = x_train[0].size();
    weights.resize(col, 0.0);
    for (int e =0; e<epochs; e++) {
        int row = x_train.size();
        for (int r=0; r<row; r++) {
            double prediction = bias;     //since y = w*x + b
            for (int c=0; c<col; c++) {
                prediction += weights[c]* x_train[r][c];
            }
            double error = prediction - y_train[r];
            for (int c=0; c<col;c++) {
                weights[c] -= learning_rate * (error) * x_train[r][c];
            }
            bias -= learning_rate * error;
        }
    }
}

double LinearRegressor::predict(vector<double> x_target) {
    double y = bias;
    for (int i =0; i<x_target.size(); i++) {
        y += weights[i]*x_target[i];
    }
    return y;
}
