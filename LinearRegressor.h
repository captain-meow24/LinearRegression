//
// Created by kanishka on 14/4/26.
//

#ifndef LINEARREGRESSOR_H
#define LINEARREGRESSOR_H

#include <vector>
#include <iostream>
using namespace std;

// a class for Loss functions
class LossFunction{
public:
    double MSE(vector<double> pred, vector<double> actual);   //calculates the mean squared error
};

pair<vector<double>,double> Gradient_Descent(vector<vector<double>> x, vector<double> y, double alpha, int epochs);
//this is the main function that updates the weights and bias values to most accurate

// The Linear Regressor
class LinearRegressor {
public:
    int epoch = 0;   //number of iterations over one data set
    double alpha = 0;   //learning rate
    vector<double> weights;   //weight of each feature
    double bias = 0;
    int features;  //number of features

    // Default epochs = 1000
    // Default learning_rate = 0.0001
    LinearRegressor(double learning_rate=0.0001 , int epoch=1000);  //constructor that sets the values of epoochs and alpha to default
    void train(vector<vector<double>> x, vector<double> y);
    vector<double> predict(vector<vector<double>> x, vector<double> pred);   //predicts the value based on most accurate weights and bias

private:
    double gettingValues(vector<double> x);
};


#endif //LINEARREGRESSOR_H
