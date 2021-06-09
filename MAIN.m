clear all;close all;clc;
K_fold = 5;
k = 3;
load('UCIData01.mat');
index_data = 3;
filname = (data_part{index_data,1});
disp(filname);
dataset = data_part{index_data,2};
curr_data = dataset;
alpha_M =  [2,4,8,16,32,64];
belta_M = [0.1:0.2:0.9];
[alpha_X,belta_X] = meshgrid(alpha_M,belta_M);
n_can = size(alpha_X,1);
n_can2 = size(alpha_X,2);

final_result = cell(n_can*n_can2,4);
acc_value = zeros(n_can,n_can2);
count = 1;
test_data = data_part{index_data,3};
test_label = data_part{index_data,4};
for i_alpha = 1:n_can
    for i_belta = 1:n_can2
        disp(['count = ',num2str(count)]);
        alpha = alpha_X(i_alpha,i_belta);
        belta = belta_X(i_alpha,i_belta);
        acc_Kfold_valide = zeros(K_fold,1);
        for i_fold = 1:K_fold
            disp(strcat('i_fold',num2str(i_fold)));
            
            valide_data =  curr_data{i_fold,1}; %选当前折为测试集
            valide_label = curr_data{i_fold,2};
            [n_valide,~] = size(valide_data);
            train_data = [];
            train_label = [];
            for j_fold = 1:K_fold
                if i_fold == j_fold
                    continue;
                end
                train_data = [train_data;curr_data{j_fold,1}];
                train_label= [train_label;curr_data{j_fold,2}];
            end
            try 
            [M] =  VirtualLMNN_1(train_data,train_label,alpha,belta); %求解马氏距离M 
            % 验证集
            preds = KNN(train_data,train_label,M,k,valide_data);
            index = find((valide_label-preds)==0);
            acc_Kfold_valide(i_fold,1) = length(index)/n_valide;
            catch
                acc_Kfold_valide(i_fold,1) = 0;
            end
        end

        curr_acc = mean(acc_Kfold_valide);
        acc_value(i_alpha,i_belta) = curr_acc;
        curr_std = std(acc_Kfold_valide);
        final_result{count,1} = acc_Kfold_valide;
        final_result{count,2} = [curr_acc,curr_std];
        final_result{count,3} = alpha;
        final_result{count,4} = belta;
        count = count +1;
    end
end

[x_index y_index]=find(acc_value==max(max(acc_value)));
x_index = x_index(1);
y_index = y_index(1);
train_data = [];
train_label = [];
for j_fold = 1:K_fold
    train_data = [train_data;curr_data{j_fold,1}];
    train_label= [train_label;curr_data{j_fold,2}];
end
[M] =  VirtualLMNN_1(train_data,train_label,alpha_X(x_index,y_index),belta_X(x_index,y_index)); %求解马氏距离M 
% 测试集
n_test = size(test_data,1);
preds_test = KNN(train_data,train_label,M,k,test_data);
index_test = find((preds_test - test_label)==0);
acc_test = length(index_test)/n_test;
filename =  strsplit(filname,'.');
save([filename{1,1},'_result_KNN1',num2str(k),'.mat'],'acc_test');
save([filename{1,1},'_result_KNN1',num2str(k),'a_',num2str(alpha_X(x_index,y_index)),'b_',num2str(belta_X(x_index,y_index)),'_all.mat'],'final_result');
        