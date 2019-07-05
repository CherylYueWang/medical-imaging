setpaths;
rng(2016);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Image files                         %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
imgf = 'MPEG7/device3-20.gif';
shape = imread(imgf);
shape = imresize(shape,[32,32]); % 32-by-32
b = zeros(2*size(shape));
b((size(b,1)/4):(size(b,1)/4)+size(shape,1)-1, ...
    (size(b,2)/4):(size(b,2)/4)+size(shape,2)-1) = shape; % 64-by-64
B = double(b);
p = size(B);
B = 0.9*B + 0.1*ones(p);
spnB = orth(B);
spnBt = orth(B');
rkB = rank(B);
m = 2;
r = 1;
B = tensor(B,p);
u = [rkB rkB];

ns = [200;800;1600];
for isim=1:3
    n = ns(isim);
    Gamma{1} = spnB; Gamma{2} = spnBt;
    Gamma0{1} = null(Gamma{1}'); Gamma0{2} = null(Gamma{2}');
    for i = 1:m
        Omega{i} = eye(u(i));
        Omega0{i} = 0.01*eye(p(i)-u(i));
        Sig{i} = Gamma{i}*Omega{i}*Gamma{i}' + Gamma0{i}*Omega0{i}*Gamma0{i}';
        Sig{i} = Sig{i}./norm(Sig{i},'fro');
        Sigsqrtm{i} = sqrtm(Sig{i});
    end
    
    Epsilon = normrnd(0,1,[r n]);
    Xn = tensor(normrnd(0,1,[p,n]));
    Xn = ttm(Xn,Sigsqrtm,1:m);
    Yn = Epsilon + squeeze(sum(sum(repmat(double(B),[1 1 n]).*double(Xn),1),2))';
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %   center the data                     %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Yn = Yn - mean(Yn);
    mux = mean(double(Xn),3);
    Xn = tensor(double(Xn)-mux(:,:,ones(n,1)));
    vecXn = reshape(double(Xn),[prod(p),n]);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %   Fit least squares                   %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [lambda, Sigx] = kroncov(Xn);
    Sigx{1} = Sigx{1}*lambda;
    Btil = kron(pinv(Sigx{2}),pinv(Sigx{1}))*vecXn*Yn'/n;
    Btil = reshape(Btil,p);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %   Fit CP                    %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [~,Bhat_cp,glmstats1,dev1] = kruskal_reg(zeros(n,1),Xn,Yn',3,'normal', ...
        'Display', 'iter', 'Replicates', 1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Fit SITPLS                            %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    uhat = u;
    [Gammahat, pghat] = TensPLS_fit(Xn,Yn,Sigx,uhat);
    Bhat_pls = kron(pghat{2},pghat{1})*vecXn*Yn'/n;
    Bhat_pls = reshape(Bhat_pls,p);
    [cv_sse, cv_u] = TensPLS_cv2d3d(Xn,Yn,40,5);
    uhat = cv_u*ones(m,1);
    [Gammahat, pghat] = TensPLS_fit(Xn,Yn,Sigx,uhat);
    Bhat_plscv = kron(pghat{2},pghat{1})*vecXn*Yn'/n;
    Bhat_plscv = reshape(Bhat_plscv,p);
    
    allBtil{isim} = -double(Btil);
    allBhat_cp{isim} = -double(Bhat_cp);
    allBhat_pls{isim} = -double(Bhat_pls);
    allBhat_plscv{isim} = -double(Bhat_plscv);
    disp(uhat);
end



i=7;
subplot(4,6,i)
subaxis(4,6,i,'SpacingVert',0.01,'SpacingHoriz',0.01,'MR',0.05);
imagesc(-double(B));
colormap(gray)
caxis([-1,0])
title('True Signal');
axis image;
axis tight;
axis ij;
set(gca,'xtick',[],'ytick',[],'layer','bottom','box','on')


vspace = 0.01; hspace = 0.01; mr = 0.05;
for isim=1:3
    i=6*isim+6;
    subplot(4,6,i)
    subaxis(4,6,i,'SpacingVert',vspace,'SpacingHoriz',hspace,'MR',mr);
    if isim==1
        text(0.05,0.5,'n=200'); axis off
    elseif isim==2
        text(0.05,0.5,'n=800'); axis off
    else
        text(0.05,0.5,'n=1600'); axis off
    end
    
    i=6*isim+2;
    subplot(4,6,i)
    subaxis(4,6,i,'SpacingVert',vspace,'SpacingHoriz',hspace,'MR',mr);
    imagesc(allBtil{isim});
    colormap(gray)
    %caxis([-1,0])
    if isim==1 
        title('OLS');
    end
    axis image;
    axis tight;
    axis ij;
    box off;
    set(gca,'xtick',[],'ytick',[],'layer','bottom')
    
    i=6*isim+3;
    subplot(4,6,i)
    subaxis(4,6,i,'SpacingVert',vspace,'SpacingHoriz',hspace,'MR',mr);
    imagesc(allBhat_cp{isim});
    colormap(gray)
    %caxis([-1,0])
    if isim==1 
        title('CP');
    end
    axis image;
    axis tight;
    axis ij;
    box off;
    set(gca,'xtick',[],'ytick',[],'layer','bottom')
    
    i=6*isim+4;
    subplot(4,6,i)
    subaxis(4,6,i,'SpacingVert',vspace,'SpacingHoriz',hspace,'MR',mr);
    imagesc((allBhat_pls{isim}));
    colormap(gray)
    %caxis([-1,0])
    if isim==1 
        title('TEPLS');
    end
    axis image;
    axis tight;
    axis ij;
    box off;
    set(gca,'xtick',[],'ytick',[],'layer','bottom')
    
    i=6*isim+5;
    subplot(4,6,i)
    subaxis(4,6,i,'SpacingVert',vspace,'SpacingHoriz',hspace,'MR',mr);
    imagesc(allBhat_plscv{isim});
    colormap(gray)
    %caxis([-1,0])
    if isim==1 
        title('TEPLS-CV');
    end
    axis image;
    axis tight;
    axis ij;
    box off;
    set(gca,'xtick',[],'ytick',[],'layer','bottom')
    
end

