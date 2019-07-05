function [Gamma, PGamma] = TensPLS_fit(Xn,Yn,SigX,u)
ss = size(Xn);
n = ss(end);
p = ss(1:end-1);
m = length(p);
SigY = cov(Yn')*(n-1)/n;

for i = 1:m
	Sinvhalf{i} = inv(sqrtm(SigX{i}));
end

Sinvhalf{m+1} = inv(sqrtm(SigY));

C = ttm(Xn,Yn,m+1)/n;

for i = 1:m
%     M = lambda*Sig{i};
%     idx = setdiff(1:m+1,i);
%     Ysn = ttm(Yn,Sinvhalf(idx(1:end-1)),idx(1:end-1));
%     idxprod = r(i)/n/prodr;
%     YsnYsn = ttt(Ysn,Ysn,idx).*idxprod;
%     U = YsnYsn.data - M;

    M = SigX{i};
    idx = setdiff(1:m+1,i);
    Ck = ttm(C,Sinvhalf(idx),idx);
    
    U = tenmat(Ck,i);
    Uk = U.data*U.data';

    Gamma{i} = EnvMU(M,Uk,u(i));
    PGamma{i} = Gamma{i}/(Gamma{i}'*SigX{i}*Gamma{i})*Gamma{i}'*SigX{i};
end

