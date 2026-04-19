#include"LinearRegressor.h"

LinearRegressor::LinearRegressor(int epoch, double learning) {
    learning_rate = learning;
    epochs = epoch;
}

void LinearRegressor::gradient_descent(vector<vector<double>> x_train, vector<double> y_train) {
    int col = x_train[0].size();
    weights.resize(col, 0.0);
    for (int e =0; e<epochs; e++) {
        int row = x_train.size();
        for (int r=0; r<row; r++) {
            double prediction =0;
            for (int c=0; c<col; c++) {
                prediction += weights[c]* x_train[r][c];
            }
            double error = y_train[r] - prediction;
            
        }
    }
}