setpaths;
rng(123456);

p = [10 10 10];
u = [2 2 2];
m = 3;
r = 1;
n = 200;

eta = tensor(rand([u r]));
for i = 1:m
    Gamma{i} = orth(rand(p(i),u(i)));
    Gamma0{i} = null(Gamma{i}');
    Omega{i} = eye(u(i));
    Omega0{i} = 0.01*eye(p(i)-u(i));
    Sig{i} = Gamma{i}*Omega{i}*Gamma{i}'+Gamma0{i}*Omega0{i}*Gamma0{i}';
    Sig{i} = Sig{i}./norm(Sig{i},'fro');
    Sigsqrtm{i} = sqrtm(Sig{i});
end
B = ttm(eta,Gamma,1:m);
SigY = 1;


nsim = 2;
pmeansqs = zeros(5,nsim);
err = zeros(5,nsim);
t_cov = 0; t_ols = 0;  t_pls = 0; t_pls_cv = 0; t_cp = 0; 
u_select = zeros(nsim,1);
for isim = 1:nsim
    disp(isim);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %   Generate data                       %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Epsilon = mvnrnd(zeros(r,1),SigY,n);
    Xn = tensor(normrnd(0,1,[p,n]));
    Xn = ttm(Xn,Sigsqrtm,1:m);
    Yn = Epsilon' + ...
        [squeeze(sum(sum(sum(repmat(double(B(:,:,:)),[1 1 1 n]).*double(Xn),1),2),3))'];
    

    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %   center the data                     %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    muy = mean(Yn')';
    Yn = Yn - muy(:,ones(n,1));
    mux = mean(double(Xn),4);
    Xn = tensor(double(Xn)-mux(:,:,:,ones(n,1)));
    vecXn = reshape(double(Xn),[prod(p),n]);
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %   Fit least squares                   %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    t0 = cputime;
    [lambda, Sigx] = kroncov(Xn);
    Sigx{1} = Sigx{1}*lambda;
    t_cov = t_cov + cputime - t0;
	t0 = cputime;
    Btil = kron(kron(pinv(Sigx{3}),pinv(Sigx{2})),pinv(Sigx{1}))*vecXn*Yn'/n;
    Btil = reshape(Btil,[p r]);
    t_ols = t_ols + cputime - t0;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %   Fit CP                    %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    t0 = cputime;
    Bhat_cp = Btil;
    for i=1:r
    [~,Bhat_cp(:,:,:,i),glmstats1,dev1] = kruskal_reg(zeros(n,1),Xn,Yn(i,:)',3,'normal', ...
        'Display', 'iter', 'Replicates', 1);
    end
    t_cp = t_cp + cputime - t0;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Fit SITPLS                            %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [cv_sse cv_u] = TensPLS_cv2d3d(Xn,Yn,10,2);
    disp(cv_u);
    u_select(isim) = cv_u;
    uhat = cv_u*ones(m,1);
    %uhat = u;
    t0 = cputime;
    [Gammahat, pghat] = TensPLS_fit(Xn,Yn,Sigx,uhat);
    Bhat_pls_cv = kron(pghat{3},kron(pghat{2},pghat{1}))*vecXn*Yn'/n;
    t_pls_cv = t_pls_cv + cputime - t0;
    Bhat_pls_cv = reshape(Bhat_pls_cv,[p r]);

    uhat = u;
    t0 = cputime;
    [Gammahat, pghat] = TensPLS_fit(Xn,Yn,Sigx,uhat);
    Bhat_pls = kron(pghat{3},kron(pghat{2},pghat{1}))*vecXn*Yn'/n;
    t_pls = t_pls + cputime - t0;
    Bhat_pls = reshape(Bhat_pls,[p r]);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %   Compare                             %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Epsilon = mvnrnd(zeros(r,1),SigY,n);
    Xn = tensor(normrnd(0,1,[p,n]));
    Xn = ttm(Xn,Sigsqrtm,1:m);
    Yn = Epsilon' + ...
    [squeeze(sum(sum(sum(repmat(double(B(:,:,:)),[1 1 1 n]).*double(Xn),1),2),3))'];
    muy = mean(Yn')';
    Yn = Yn - muy(:,ones(n,1));
    mux = mean(double(Xn),4);
    Xn = tensor(double(Xn)-mux(:,:,:,ones(n,1)));
    
    pmeansqs(1,isim)= PMSE(Xn,Yn,double(B));
    pmeansqs(2,isim)= PMSE(Xn,Yn,Btil);
    pmeansqs(3,isim)= PMSE(Xn,Yn,Bhat_cp);
    pmeansqs(4,isim)= PMSE(Xn,Yn,Bhat_pls);
    pmeansqs(5,isim)= PMSE(Xn,Yn,Bhat_pls_cv);
    err(1,isim)= BErr(B,B);
    err(2,isim)= BErr(B,Btil);
    err(3,isim)= BErr(B,Bhat_cp);
    err(4,isim)= BErr(B,Bhat_pls);
    err(5,isim)= BErr(B,Bhat_pls_cv);
end


mean(err(4:end,:)')
std(err(4:end,:)')./sqrt(nsim)
sum(u_select==1)
mean(pmeansqs')
std(pmeansqs')./sqrt(nsim)
mean(err')
std(err')./sqrt(nsim)
mean(err'./prod(p))
std(err'./prod(p))./sqrt(nsim)



