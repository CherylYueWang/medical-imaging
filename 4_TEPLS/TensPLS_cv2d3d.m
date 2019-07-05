function [cv_sse u] = TensPLS_cv2d3d(X0,Y0,maxdim,nfolds)
ss = size(X0);
n = ss(end);
p = ss(1:end-1);
r = size(Y0,1);
m = length(p);
vecX0 = reshape(double(X0),[prod(p),n]);
idx = randperm(n);
Ntest = floor(n/nfolds);
Ntrain = n - Ntest;
cv_sse = zeros(maxdim,1);

for i=1:nfolds
    testid = [1:Ntest] + (i-1)*Ntest;
    testid = idx(testid);
    Ytrain = Y0; vecXtrain = vecX0; vecXtrain(:,testid) = []; Ytrain(:,testid) = [];
    mu_vecX = mean(vecXtrain,2); mu_Y = mean(Ytrain,2);
    Ytrain = Ytrain - mu_Y(:,ones(Ntrain,1)); vecXtrain = vecXtrain - mu_vecX(:,ones(Ntrain,1));
    Xtrain = tensor(reshape(vecXtrain,[p, Ntrain]));
    Ytest = Y0(:,testid); vecXtest = vecX0(:,testid);
    Ytest = Ytest - mu_Y(:,ones(Ntest,1)); vecXtest = vecXtest - mu_vecX(:,ones(Ntest,1));
    Xtest = tensor(reshape(vecXtest,[p,Ntest]));
    
    %%%%%%%%%%% Fit TPLS
    [lambda, SigX] = kroncov(Xtrain);
    SigX{1} = lambda*SigX{1};
    [Gamma, PGamma] = TensPLS_fit(Xtrain,Ytrain,SigX,maxdim*ones(m,1));
    for k=1:maxdim
        for j=1:m
            if k==p(j)
                Ghat{j} = eye(p(j));
            else
                Gtmp = Gamma{j};
                Ghat{j} = Gtmp(:,1:k);
                PGamma{j} = Ghat{j}/(Ghat{j}'*SigX{j}*Ghat{j})*Ghat{j}';
            end
            if m==2
                Bhat_pls = kron(PGamma{2},PGamma{1})*vecXtrain*Ytrain'/Ntrain;
            elseif m==3
                Bhat_pls = kron(PGamma{3},kron(PGamma{2},PGamma{1}))*vecXtrain*Ytrain'/Ntrain;
            end
        end
        ehat = Bhat_pls'*vecXtest - Ytest;
        cv_sse(k) = cv_sse(k) + trace(ehat*ehat');
    end
end

[mincv, u] = min(cv_sse);
