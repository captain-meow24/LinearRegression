#include "LinearRegressor.h"
#include <iostream>
using namespace std;
int main() {
    vector<vector<double>> x_train = {
        {1, 1}, {2, 1}, {3, 2}, {4, 2}, {5, 3},
        {6, 3}, {7, 4}, {8, 4}, {9, 5}, {10, 5},
        {2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 7}
    };

    vector<double> y_train = {
        10.1, 12.0, 17.2, 19.1, 24.0,
        26.2, 31.1, 33.0, 38.3, 40.2,
        18.0, 23.2, 28.1, 33.0, 38.4
    };
    vector<vector<double>> x_test = {
        {3, 1},
        {4, 2},
        {6, 3},
        {7, 5},
        {9, 4}
    };

    vector<double> y_test = {
        14.0,
        19.0,
        26.0,
        34.0,
        35.0
    };
    LinearRegressor model;
    model.gradient_descent(x_train, y_train);
    double model_error = model.accuracy(x_test, y_test);
    cout<< model_error<<endl;
    vector<double> x_pred = {3,1};
    double predicted = model.predict(x_pred);
    cout<< predicted;

    return 0;
}