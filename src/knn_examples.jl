# package and functions ------------------------------
using RDatasets, CSV, DataFrames, BenchmarkTools

include("src/knn_functions.jl")


# check datasets --------------------------------------
datasets_tb = RDatasets.datasets("datasets")
ISLR_tb = RDatasets.datasets("ISLR")
datasets_tb[10:30, 2:3]


# knn classifier -------------------------------------
df = dataset("datasets", "iris");
X = Array(df[:, 1:end-1]);
y = Array(df[:, end]);

K = 5
@time model = KnnClassifier(X, y, K);

predict(model, rand(4))
predict(model, rand(10, 4))

y_pred = predict(model, X)
sum(y .== y_pred)

@benchmark begin
    model = KnnClassifier(X, y, 5);
    predict(model, rand(4))
end


# knn regression -------------------------------------
auto = dataset("ISLR", "Auto")      # names(auto)

X = Array(auto[:, 1:4]);
y = Array(auto[:, 5]);

K = 5
@time model = KnnRegression(X, y, K);

predict(model, rand(4))
predict(model, rand(10, 4)*1000)

y_pred = predict(model, X);

norm(y - y_pred)    # mse
mean(y - y_pred)    # media dos erros

@benchmark begin
    model = KnnRegression(X, y, 5);
    predict(model, rand(4))
end