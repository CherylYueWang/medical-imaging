function mse = PMSE(Xn,Yn,Bhat)
% Xn: p*n; Yn: r*n; Bhat: p*r
ss = size(Xn);
n = ss(end);
p = ss(1:end-1);
r = size(Yn,1);
m = length(p);
Yhat = reshape(double(Bhat),[prod(p) r])'*reshape(double(Xn),[prod(p),n]);
Epsilon = Yn - Yhat;    % r-times-n
mse = trace(double(Epsilon*Epsilon'))/n;