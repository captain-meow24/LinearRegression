//
// Created by kanishka on 14/4/26.
//

#include "LinearRegressor.h"
#include <iomanip>

int main() {
    // Making an object of Linear Regression model
    LinearRegressor model(0.0001,30000);
    // Making an object of the Loss Function model to print the amount of loss/estimate of accuracy
    LossFunction LF;
    //setting data
    vector<vector<double>> x_train {
        {10, 2}, {20, 3}, {30, 4}, {40, 5}, {50, 6},
        {60, 7}, {70, 8}, {80, 9}, {90, 10}, {100, 11},
        {15, 2}, {25, 3}, {35, 4}, {45, 5}, {55, 6},
        {65, 7}, {75, 8}, {85, 9}, {95, 10}, {105, 11}
    };
    vector<double> y_train {
        9, 16, 23, 30, 37,
        44, 51, 58, 65, 72,
        11.5, 18.5, 25.5, 32.5, 39.5,
        46.5, 53.5, 60.5, 67.5, 74.5
    };
    vector<vector<double>> x_test {
        {12, 2}, {28, 4}, {46, 6}, {68, 8}, {90, 10}
    };
    vector<double> y_test {
        10, 22, 35, 50, 65
    };

    // Training the model with .train() method
    model.train(x_train, y_train);

    // Predicting using .predict() method
    vector<double> y_pred;
    vector<double> pred = model.predict(x_test, y_pred);


    // Printing the predicted value alongside actual value for comparison
    cout<<"Predicted Value\t\t"<<"Actual Value"<<endl;
    for(int i = 0; i<y_test.size(); i++){
        cout<<pred.at(i)<<"\t\t\t";
        cout<<(y_test.at(i))<<endl;
    }

    // Calculating Loss
    double loss = LF.MSE(pred, y_test);

    // Printing Loss
    cout << fixed << setprecision(9);
    cout<<endl<<"Loss is: "<<loss<<endl;

    return 0;
}