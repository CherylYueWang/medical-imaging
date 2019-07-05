function [lambda S] = kroncov(Tn)
ss = size(Tn);
n = ss(end);
r = ss(1:end-1);
m = length(r);
prodr = prod(r);


%%%%%%%%%% initialization %%%%%%%%%%
lambda = 1;
for i = 1:m
    S{i} = eye(r(i));
    Sinvhalf = S;
end

%%%%%%%%% iteration %%%%%%%%%%%%%%%
for isim = 1:5
%    conv = 0;
    for i = 1:m
        Si0 = S{i};
        idx = setdiff(1:m+1,i);
        Tsn = ttm(Tn,Sinvhalf(idx(1:end-1)),idx(1:end-1));
        idxprod = r(i)/n/prodr;
        TsnTsn = ttt(Tsn,Tsn,idx).*idxprod;
        S{i} = TsnTsn.data;
        S{i} = S{i}./norm(S{i},'fro');
        %S{i} = TsnTsn.data;
        Sinvhalf{i} = inv(sqrtm(S{i}));
%         if norm(Si0-S{i},'fro')<etol
%             conv = conv+1;
%         end
    end
    Tsn = ttm(Tn,Sinvhalf,1:m);
    lambda = (norm(Tsn))^2./prod([r n]);
%     if conv==m
%         break;
%     end
end




