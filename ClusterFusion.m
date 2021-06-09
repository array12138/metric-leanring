function [U_matrix_new,Cluster_new]= ClusterFusion(U_matrix,Cluster)
% Fusion the virtual point
% U_matrix:    n*d virtual matrix
% Cluster£º    1*n label vector
% Ratio:       Threshold, the distance between classes is less than this threshold, then fusion
% Weights:     a weight, if x_j is x_i neighbor, w_ij = 1,else w_ij = 0
    difflabel = unique(Cluster);
    n_Cluster = length(difflabel);
    Dist = zeros(n_Cluster,n_Cluster);
    Dist(:) = inf;
    cluster_partition_index = zeros(n_Cluster,1); % 
    for i = 1:(n_Cluster-1)
        cluster_partition_index(i) = difflabel(i);
        oneClass = find(Cluster == difflabel(i));
        for j = i+1:n_Cluster
            twoClass = find(Cluster == difflabel(j));
            temp = U_matrix(oneClass(1),:) -  U_matrix(twoClass(1),:);
            Dist(i,j) = temp* temp';
        end
    end
    cluster_partition_index(n_Cluster) = difflabel(n_Cluster); % 
    RATIO = 1.02;
    min_value = min(Dist(:));
    [row,col]= find(Dist<=min_value * RATIO);
    % 3 Recode the index of the class to find fused nodes with breadth-first search
    all_value = unique([row,col]);
    newrow = zeros(length(row),1);
    newcol = zeros(length(row),1);
    for i = 1:length(row)
         index = find(all_value == row(i));
         newrow(i) = index;
         index = find(all_value == col(i));
         newcol(i) = index;
    end
    Graph_matrix = graph(newrow,newcol); % Calculating the nearest neighbor graph
    index_all = zeros(length(all_value),1); % Records whether the node has been accessed
    temp_label = Cluster;
    % 4 Perform a BFS search on each node
    for i = 1:length(all_value)
        node  = bfsearch(Graph_matrix,i); % Breadth-first search for node i
        temp_mu = zeros(1,size(U_matrix,2));
        count = 0;
        if sum(index_all(node(:))) ==0 % If the node has not been used
             for j = 1:length(node) % The connectivity component targeted to this i_node or for all clusters to be fused
                 oneclass = find(Cluster == Cluster(cluster_partition_index(all_value(node(j))))); % Cj fused class
                 count  = count + length(oneclass);
                 if isempty(oneclass)
                    disp('hahahah');
                 end 
                 temp_mu = temp_mu + length(oneclass)* U_matrix(oneClass(1),:);
             end
             all_sam = [];
             for j = 1:length(node)  % The connectivity component targeted to this i_node
                oneclass = find(Cluster == Cluster(cluster_partition_index(all_value(node(j))))); % Cj fused class
                all_sam = [all_sam;oneclass];
                for k = 1:length(oneclass)
                    U_matrix(oneclass(k),:) = temp_mu/count;
                end
             end
             temp_label(all_sam(:)) = min(Cluster(all_sam(:)));
        end
        index_all(node(:)) = 1;
    end
    U_matrix_new = U_matrix;
    Cluster_new = temp_label;
end