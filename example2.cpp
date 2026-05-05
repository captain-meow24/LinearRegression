//
// Created by kanishka on 5/5/26.
//
#include "LinearRegressor.h"
#include <iostream>
using namespace std;

int main() {
    vector<vector<double>> x_train = {
        {1,2},{2,3},{3,4},{4,5},{5,6},
        {6,7},{7,8},{8,9},{9,10},{10,11},
        {11,12},{12,13},{13,14},{14,15},{15,16},
        {16,17},{17,18},{18,19},{19,20},{20,21},
        {2,5},{3,6},{4,7},{5,8},{6,9},
        {7,10},{8,11},{9,12},{10,13},{11,14}
    };

    vector<double> y_train = {
        15.2,20.1,25.0,29.9,35.1,
        40.0,45.2,50.1,55.0,60.2,
        65.1,70.0,75.2,80.1,85.0,
        90.2,95.1,100.0,105.2,110.1,
        22.0,27.1,32.0,37.2,42.1,
        47.0,52.2,57.1,62.0,67.2
    };

    vector<vector<double>> x_test = {
        {3,4},
        {6,7},
        {10,11},
        {15,16},
        {18,19}
    };

    vector<double> y_test = {
        25.0,
        40.0,
        60.0,
        85.0,
        100.0
    };

    LinearRegressor model;
    model.gradient_descent(x_train, y_train);

    double model_error = model.accuracy(x_test, y_test);
    cout << "model error: " << model_error << endl;

    vector<double> x_pred = {12,13};
    double predicted = model.predict(x_pred);
    cout << "prediction: " << predicted;

    return 0;
}