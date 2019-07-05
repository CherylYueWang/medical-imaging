function M = gen_arcorr(p,rho,sigm)
% p-by-p covariance, sigm*AR(rho)
M = eye(p);
for i=1:p
    for j=1:p
        M(i,j) = rho^(abs(i-j));
    end
end

M = sigm*M;
