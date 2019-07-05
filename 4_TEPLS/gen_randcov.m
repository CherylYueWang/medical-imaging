function M = gen_randcov(p)
% p-by-p covariance
M = rand(p,p);
M = M'*M;
