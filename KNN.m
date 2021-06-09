function preds = KNN(X,y, M, k, Xt)
% function preds = KNN(y, X, M, k, Xt)
% X :训练集
% y :训练集的标签
% M ：距离的度量矩阵
% k :knn的个数
% Xt:测试集

add1 = 0;
if (min(y) == 0),
    y = y + 1;
    add1 = 1;
end
[n,m] = size(X);
[nt, m] = size(Xt);
K = (X*M*Xt');
l = zeros(n,1);
for (i=1:n),
    l(i) = (X(i,:)*M*X(i,:)');
end

lt = zeros(nt,1);
for i=1:nt
    lt(i) = (Xt(i,:)*M*Xt(i,:)');
end

D = zeros(n, nt);
for i=1:n
    for j=1:nt
        D(i,j) = l(i) + lt(j) - 2 * K(i, j);
    end
end

[V, Inds] = sort(D);

preds = zeros(nt,1);
for i=1:nt
    counts = [];
    for j=1:k       
        if (y(Inds(j,i)) > length(counts)),
            counts(y(Inds(j,i))) = 1;
        else
            counts(y(Inds(j,i))) = counts(y(Inds(j,i))) + 1;
        end
    end
    [v, preds(i)] = max(counts);
end
if (add1 == 1),
    preds = preds - 1;
end