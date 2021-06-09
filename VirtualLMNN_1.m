function [M_matrix,U_matrix] = VirtualLMNN_1(data,label,k_neighbor,Mu)
%label标签索引必须得按1，2，3，4进行排列
[n_sam,n_fea] = size(data); % sample num, feature num
M_matrix = eye(n_fea,n_fea);           % orign B,metric matrix
L_matrix = eye(n_fea,n_fea);
EIG_matrix = Mu * eye(n_fea,n_fea);
U_matrix = data; % origin
% RATIO = 1e-2;
% Beta = Beta * Mu;

lablist = unique(label);
K = length(lablist);

newLabel = zeros(n_sam,1);

for i = 1:K
    oneClass = find(lablist(i)==label);
    newLabel(oneClass(:)) = i;
end
label_matrix = false(n_sam, K);
label_matrix(sub2ind(size(label_matrix), (1:length(label))', newLabel)) = true;
same_label = logical(double(label_matrix) * double(label_matrix'));

GAMMA = 0;
X = data;
sum_X = sum(X .^ 2, 2);
DD = bsxfun(@plus, sum_X, bsxfun(@plus, sum_X', -2 * (X * X')));
DD_old = DD;
DD(~same_label) = Inf;DD(1:n_sam + 1:end) = Inf;
[~, targets_ind] = sort(DD, 2, 'ascend');
% k_neighbor = 3;   % k_neighbor number
targets_ind = targets_ind(:,1:k_neighbor);
targets = false(n_sam, n_sam);
targets(sub2ind([n_sam n_sam], vec(repmat((1:n_sam)', [1 k_neighbor])), vec(targets_ind))) = true;
Weights = zeros(n_sam,n_sam);
for i = 1:n_sam
    for j = 1:n_sam
        if targets(i,j)
            Weights(i,j) = exp(-GAMMA * DD_old(i,j));
        end
    end
end

is_update =false;
[U_matrix] = compute_Ui(data,label,U_matrix,M_matrix,L_matrix,EIG_matrix,100,Mu,Weights);
for i = 1:n_sam
    temp = (data(i,:)-U_matrix(i,:)) * (data(i,:)-U_matrix(i,:))';
    if temp>=1e-10
        is_update = true;
        break;
    end
end
if ~is_update
    disp('U_matrix is not update');
    return;
end
[M_matrix,~,~] = my_lmnn(data,U_matrix,M_matrix,label,Mu);
Cluster = 1:size(data,1);
[U_matrix,Cluster] = ClusterFusion(U_matrix,Cluster);


end
function [U_matrix_new] = compute_Ui(Data,label,U_matrix,M_matrix,L_matrix,EIG_matrix,Beta,Mu,Weights)
% compute the virtual matrix n*d 计算虚拟点矩阵
% Data:              a n*d origin dataset
% label:             a n*1 label column vector 
% U_matrix:          a n*d virtual dataset
% M_matrix:          a d*d metric matrix
% L_matrix：         a feature vector matrix of M_matrix 
% EIG_matrix：       a Eigenvalues matrix of M_matrix
% Beta,Mu :      regularization parameter
% return U_matrix_new 
    [n_sam,n_fea] = size(Data); 
    U_matrix_new = zeros(n_sam,n_fea);
    for i = 1:n_sam
        Pl_index = find(label(i) ~= label); % xi and xl unsimilar label        
        Ck_index = find(label(i) == label); % C_k: xi class
        nCk = length(Ck_index);                 % |C_k| length
        x_i = Data(i,:);
        u_i = U_matrix(i,:);
        dist_ii = (u_i - x_i) * M_matrix * (u_i - x_i)';
        
        G_index = [];
        sum_weight = 0;
        for j = 1:nCk
            temp = U_matrix(Ck_index(j),:) - u_i;
            if sum(abs(temp))>=1e-10 %如果两个u_j与u_i不相等，不能写为U_matrix(Ck_index(j),:)~=u_i
                G_index = [G_index;Ck_index(j)];
                sum_weight = sum_weight + Weights(i,Ck_index(j));
            end
        end
        nG = size(G_index,1);
 
        
        EIG_flag = zeros(n_fea,n_fea);
        diag_index = sub2ind(size(EIG_matrix),1:n_fea,1:n_fea);
        EIG_flag(diag_index) = 1./(EIG_matrix(diag_index) + Beta * sum_weight * ones(1,n_fea));
        part_left = L_matrix * EIG_flag * L_matrix';
        
        part_right1 = Mu * M_matrix * x_i';
        u_j_avg = zeros(n_fea,1);
        for j = 1:nG
            u_j = U_matrix(G_index(j),:);
            u_j_avg = u_j_avg + Weights(i,G_index(j))* u_j';
        end
        part_right2 = u_j_avg .* Beta;
        
        part_right3 = zeros(n_fea,1);
        for m = 1: length(Pl_index)
            x_m =  Data(Pl_index(m),:);
            dist_im = (u_i - x_m) * M_matrix * (u_i - x_m)';
            if 1 + dist_ii >= dist_im
                part_right3 = part_right3 + (1-Mu) * M_matrix * (x_m - x_i)';
            end
        end
        newu_i = part_left * (part_right1 + part_right2 - part_right3);
        U_matrix_new(i,:) = newu_i';
    end
end
function x = vec(x)
    x = x(:);
end



