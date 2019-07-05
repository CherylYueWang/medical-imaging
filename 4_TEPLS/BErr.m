function Err = BErr(B,Bhat)

p = size(B);
m = length(p);
Err = reshape(double(Bhat)-double(B),[prod(p) 1]);
Err = sqrt(Err'*Err);