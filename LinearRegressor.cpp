#include"LinearRegressor.h"
#include<riscv_vector.h>

LinearRegressor::LinearRegressor(int epoch, double learning) {
    learning_rate = learning;
    epochs = epoch;
}

void LinearRegressor::gradient_descent(vector<vector<double>>& x_train, vector<double>& y_train) {
    int col = x_train[0].size();
    weights.resize(col, 0.0);
    int row = x_train.size();
    for (int e = 0; e<epochs; e++) {
        for (int r=0; r<row; r++) {
            double prediction = bias;     //since y = w*x + b
            /*
             vector support for processing 4 features at a time
             */
            for (int c=0; c<col; c++) {
                prediction += weights[c]* x_train[r][c];
            }
            double error = prediction - y_train[r];
            for (int c=0; c<col;c++) {    //vector support here to change 4 weights at a time
                weights[c] -= learning_rate * (error) * x_train[r][c];
            }
            bias -= learning_rate * error;
        }
    }
}

double LinearRegressor::predict(vector<double>& x_target) {
    double y = bias;
   // for (int i =0; i<x_target.size(); i++) {  //vector support here to multiply multiple weights with their target x at a time
     //   y += weights[i]*x_target[i];
   // }
    int h =0;
    size_t tr = x_target.size();
    while (h<tr) {
        size_t v = __riscv_vsetvl_e64m1(tr -h);
        vfloat64m1_t va = __riscv_vle64_v_f64m1(&x_target[h], v);   //loading features into va
        vfloat64m1_t vb = __riscv_vle64_v_f64m1(&weights[h], v);      //loading weights into vb
        vfloat64m1_t vmul = __riscv_vfmul_vv_f64m1(va, vb, v);    //multiplying the two vectors
        vfloat64m1_t vsum = __riscv_vfredusum_vs_f64m1_f64m1(
           vmul, __riscv_vfmv_v_f_f64m1(0.0, v),  v);
        double partial = __riscv_vfmv_f_s_f64m1_f64(vsum);

        y += partial;

        h += v;
    }

    return y;
}
double LinearRegressor::MSE(vector<double> &predicted, vector<double> &y_test) {
    //perform vector subtraction
    double mse = 0;
    int vecsize = predicted.size();
    int i =0;
    while (i<vecsize) {
        size_t vl = __riscv_vsetvl_e64m1(vecsize-i);
        vfloat64m1_t va = __riscv_vle64_v_f64m1(&predicted[i], vl);   //loading predicted into va
        vfloat64m1_t vb = __riscv_vle64_v_f64m1(&y_test[i], vl);      //loading test y into vb
        vfloat64m1_t vdiff = __riscv_vfsub_vv_f64m1(va,vb,vl);     //calculating difference
        vfloat64m1_t vsq = __riscv_vfmul_vv_f64m1(vdiff, vdiff, vl);
        vfloat64m1_t vsum = __riscv_vfredusum_vs_f64m1_f64m1(
           vsq, __riscv_vfmv_v_f_f64m1(0.0, vl),  vl);     //when we add a and b, we do a = 0 + b,
                                                      // so here we need a 0 vector that we could add to our diff to calculate the sum
        double partial = __riscv_vfmv_f_s_f64m1_f64(vsum);

        mse += partial;
        i += vl;
    }
    return mse/vecsize;
}

double LinearRegressor::accuracy(vector<vector<double> > &x_test, vector<double> &y_test) {
    double error = 0.0;
    double diff = 0.0;
    int samples = y_test.size();
    vector<double> predicted;
    for (int i=0; i<x_test.size(); i++){ // optimize using vector instr
        diff = predict(x_test[i]);
        predicted.push_back(diff);
    }
    error = MSE(predicted, y_test);
    return error/samples;
}

