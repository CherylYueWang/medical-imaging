function Gamma = EnvMU(M,U,m)
p = size(U,1);
W = zeros(p,m+1);
for k=1:m
    Wk = W(:,1:k);
    Ek = M*Wk;
    QEk = eye(p) - Ek*pinv(Ek'*Ek)*Ek';
    [W(:,k+1),~] = eigs(QEk*U*QEk,1);
end
Gamma = orth(W(:,2:end));
