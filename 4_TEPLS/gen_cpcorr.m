function M = gen_cpcorr(p,rho,sigm)
% p-by-p covariance, sigm*CP(rho)
M = eye(p);
for i=1:p
    for j=1:p
        if i~=j
            M(i,j) = rho;
        end
    end
end

M = sigm*M;
