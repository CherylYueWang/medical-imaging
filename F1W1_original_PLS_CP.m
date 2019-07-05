%imgf = 'case_study/train/A01_C1_F1_s01_w2.TIF';
%shape = imread(imgf);
F1W1_label = importdata("case_study/F1_W1_label.csv");
image_name_cell = F1W1_label.textdata(2:end,2);
imgf = tensor();
for image_idx = 1:400
    disp(image_idx);
    image_name = image_name_cell(image_idx);
    formatSpec = 'case_study/train/%s';
    file_name = sprintf(formatSpec,string(image_name));
    shape = imread(file_name);
    imgf(:,:,image_idx) = imresize(shape,[65,87]);
end
train_idx = csvread('case_study/F1W1_train_idx.csv',1,1);
M_train = imgf(:,:,train_idx(1:300,:));
test_idx = csvread('case_study/F1W1_test_idx.csv',1,1);
M_test = imgf(:,:,test_idx);
y = F1W1_label.data(:,1);
y_train = y(train_idx(1:300,:));
y_test = y(test_idx);

X_train = zeros(300,1);
X_test = zeros(100,1);
% CP 
% disp('cp')
% train_mse = [];
% test_mse = [];
% rank = 4;
% for r = 1:rank
%     [beta0,beta,~] = kruskal_reg(X_train,M_train,y_train,r,'normal');
%     y_train_hat = double(ttt(tensor(beta), M_train, 1:2));
%     mse_train = mean((y_train_hat - y_train).^2);
%     train_mse(r) = mse_train;
%     y_test_hat = double(ttt(tensor(beta), M_test, 1:2));
%     mse_test = mean((y_test_hat - y_test).^2);
%     test_mse(r) = mse_test;
% end

% pls
p = size(M_train(:,:,1));
disp(p);
m = 2;
y_train = y_train - mean(y_train);
muM = mean(double(M_train),3);
M_train = tensor(double(M_train)-muM(:,:,ones(300,1)));
vecM_train = reshape(double(M_train),[prod(p),300]);
% least square
[lambda, Sigx] = kroncov(M_train);
Sigx{1} = Sigx{1}*lambda;

disp("pls");
[cv_sse, cv_u] = TensPLS_cv2d3d(M_train,y_train',40,5);
uhat = cv_u*ones(m,1);
[Gammahat, pghat] = TensPLS_fit(M_train,y_train',Sigx,uhat);
n = 300;
Bhat_plscv = kron(pghat{2},pghat{1})*vecM_train*y_train/n;
Bhat_plscv = reshape(Bhat_plscv,p);
mse_train = PMSE(M_train,y_train',Bhat_plscv);
mse_test = PMSE(M_test,y_test',Bhat_plscv);