This is a c++ based Library for Linear Regresson optimized using RISC-V vector instructions

# How to use :
When in need of running a machine learning model (linear regression), just copy paste LinearRegression.h and LinearRegression.cpp to your project.

# Uses :
The model has to be trained on training data (in the form of vectors), the features being a 2D matrix and the target being a single value for each row of data.
Use gradient_descent() to train the data
predict() will output prediction for a given set of data (only after training has been performed)
accuracy() will output avrage error, has to be provided testing data.

# Example run ;
 
```cpp
LinearRegressor model;
model.gradient_descent(x_train, y_train);

double model_error = model.accuracy(x_test, y_test);
cout << "model error: " << model_error << endl;

vector<double> x_pred = {3, 1};
double predicted = model.predict(x_pred);

cout << "prediction: " << predicted;
