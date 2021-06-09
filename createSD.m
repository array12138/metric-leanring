function [D,S] = createSD( X,Y )
%Create Similarity, Disimilarity and X S matrix of Xings paper
[n,d]=size(X);
D = zeros(n);
S = zeros(n);
for i=1:(n-1)
    for j=i+1:n
        if (Y(i) == Y(j)) %% They are similar
            S(i,j)=1;
        else    %% Dissimilar
            D(i,j)=1; %% Dissimilarity pairs
        end
    end
end
end