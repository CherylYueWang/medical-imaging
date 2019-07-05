clear
% load csv
% train
train_image_mpca = csvread('mpca_F1_W1_train.csv',1,1);
n_train = 400;
y_train = train_image_mpca(:,1); % outcome
X_train = zeros(n_train,1);% regular covariate
%X_test = zeros(n_test,1);% regular covariate
M_train = tensor(); % tensor covariate
for i = 1:n_train
    raw_vector = train_image_mpca(i, 2:end);
    new_matrix = reshape(raw_vector,[10,15]);
    M_train(:,:,i) = new_matrix;
end
% test
test_image_mpca = csvread('mpca_F1_W1_test.csv',1,1);
n_test = 200;
y_test = test_image_mpca(:,1); % outcome
X_test = zeros(n_test,1);% regular covariate
M_test = tensor(); % tensor covariate
for i = 1:n_test
    raw_vector = test_image_mpca(i, 2:end);
    new_matrix = reshape(raw_vector,[10,15]);
    M_test(:,:,i) = new_matrix;
end


% direct regression
train_mse = [];
test_mse = [];
rank = 1;
for r = 1:rank
[beta0,beta,~] = kruskal_reg(X_train,M_train,y_train,r,'normal');
y_train_hat = double(ttt(tensor(beta), M_train, 1:2));
mse_train = mean((y_train_hat - y_train).^2);
train_mse(r) = mse_train;
y_test_hat = double(ttt(tensor(beta), M_test, 1:2));
mse_test = mean((y_test_hat - y_test).^2);
test_mse(r) = mse_test;
end
csvwrite('predict_y_F1W1.csv',y_test_hat)