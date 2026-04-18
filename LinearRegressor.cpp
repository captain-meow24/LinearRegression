//
// Created by kanishka on 14/4/26.
//
#include "LinearRegressor.h"
#include <iostream>
#include <cmath>
#include <vector>
using namespace std;


double LossFunction::MSE(vector<double> pred, vector<double> actual){

    // m stores number of datapoints
    double m = actual.size();
    // Initializing the loss
    double loss = 0.0;
    // Iterating m times
    for(int i=0; i<m; i++){
        // Adding the squared errors
        loss += pow(actual[i] - pred[i], 2);
    }
    double FinalLoss = loss/(2*m);
    return FinalLoss;

}


// this function minimizes the error and finds the best weights and bias.
//Takes in x, y, number of datapoints, number of features and the learning rate and the number of epochs

pair<vector<double>,double> Gradient_Descent(vector<vector<double>> x, vector<double> y, double alpha, int epochs){
    // m stores number of DataPoints
    int m = x.size();
    // n stores number of features
    int n = x[0].size();
    double bias = 0;
    vector<double> weights(n, 0.0);
    for (int z=0;z<epochs; z++) {   //running loops over the whole dataset
        //Setting the updated weight and updated bias to 0 after every iteration because we find fresh error each iteration and dont want any redundant value
        double updated_bias = 0;
        vector<double> updated_weights(n, 0.0);
        // Loop for adding the derivated error for all the datapoints
        for (int i=0; i<m; i++){
            // Calculating error
            double prediction = bias;
            for (int k=0; k<n; k++){
                prediction += weights[k] * x[i][k];
            }
            double error = y[i] - prediction;
            // Derivation
            for (int k=0; k<n; k++){
                updated_weights[k] += -(1.0/m)*(x[i][k])*error;  //this is a derivation of MSE with respect to weights
                //this will give the direction and rate at which the error will change with respect to weights
            }
            updated_bias = updated_bias + (-(1.0/m)*error);  //derivation of MSE with respect to bias
        }
        // Updating the weights and bias
        for (int i=0; i<n; i++){
            weights[i] = weights[i] - updated_weights[i]*alpha;   //we are subtracting here because if the error changes negatively with the
            //value of the feature/bias, it will be positive and vice versa
        }
        bias = bias - updated_bias*alpha;
    }
    return make_pair(weights, bias);
}

// Takes in the learning rate and number of epochs
LinearRegressor::LinearRegressor(double learning_rate, int epoch){
    //constructor to assing values to alpha and epoch
    this->alpha = learning_rate;
    this->epoch = epoch;
}

// Used to train the model and set the weights and bias
// It uses the function Gradient_Descent
void LinearRegressor::train(vector<vector<double>> x, vector<double> y){
    int sz = x.size();
    int features = x[0].size();
    pair<vector<double>,double> WeightsAndBias = Gradient_Descent(x,y,alpha,epoch);
    this->weights = WeightsAndBias.first;
    this->bias = WeightsAndBias.second;
    this->features = features;
}

// This is used to predict the set of x values
vector<double> LinearRegressor::predict(vector<vector<double>> x, vector<double> pred){
    int sizze = x.size();  //input rows
    // Loop for getting all the predicted value individually
    for (int i=0; i<sizze; i++){
        pred.push_back(gettingValues(x[i]));
    }
    return pred;   //the vector that contains all predicted values;
}

// Used to predict a single y value for x
double LinearRegressor::gettingValues(vector<double> x){
    // Initialized y as bias so that the bias is already added to the target
    double y = bias;
    for (int i = 0; i < features; i++) {
        y += weights[i] * x[i];
    }
    return y;
}



