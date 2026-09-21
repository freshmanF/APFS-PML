clc;
clear;
data_name = './data/image.mat';
load(data_name);

partialRate  =1.0;
feaNoiseRate = 0.3;

evamode = 1; 
HyperPara.ablationx = 0;
HyperPara.ablationy = 0;
HyperPara.ablationd = 0;

model_name = 'APFS-PML';

[train_data, settings]=mapminmax(train_data');
test_data=mapminmax('apply',test_data',settings);
train_data=train_data';
test_data=test_data';
train_data(isnan(train_data))=0;
test_data(isnan(test_data))=0;

train_target(train_target==-1) = 0;
test_target(test_target==-1) = 0;
disp(size(train_target'))
PL = getPartialLabel(train_target', partialRate, 0);
disp(size(PL))
[num_instance, num_feature] = size(train_data);
[num_label, ~]   = size(train_target);

clear_data = [train_data; test_data];
claer_train_data = train_data;
clear_test_data = test_data;
noisy_data = FeatureNoise(clear_data, feaNoiseRate);
train_data = noisy_data(1:num_instance,:);
test_data = noisy_data(num_instance+1:end,:);

HyperPara.ins_num = num_instance;
HyperPara.class = size(train_target, 1);
HyperPara.k = 20;
HyperPara.closedform = 0;
HyperPara.uselip = 0;
fprintf('Running--');

HyperPara.alpha   = 8*num_instance/num_feature;     
HyperPara.beta    = 0.1;                                                    
HyperPara.maxIter = 100;
HyperPara.minLossMargin = 0.01;
HyperPara.lambda = 1; 
HyperPara.eta=0.001; 
HyperPara.gamma   = 1;
HyperPara.k_initial = 25; 
HyperPara.omega = 0.67;
HyperPara.mu = 1e-3; 
HyperPara.admm_rho = 1;

[W, A, Distribution] = APFS_PML_Optimization(train_data, PL, HyperPara);

[dumb, index] = sort(sum(W.*W,2),'descend');
index = index( index <= size(train_data,2) ); 
Num = 10;
Smooth = 1;
PL(PL==0) = -1;

if evamode == 1
    NumOfInterval = 25;
    Dt = ceil(num_feature*0.5);  
    step = ceil(Dt/NumOfInterval);
    
    iterResult = zeros(15, NumOfInterval);
d_list = zeros(1, NumOfInterval);  
cnt = 0;
for d = 1:step:((NumOfInterval-1)*step+1)
    order = (d-1)/step+1;
    cnt = cnt + 1;
    d_list(order) = d;           
    f = index(1:d);
    [Prior,PriorN,Cond,CondN]=MLKNN_train(train_data(:,f), PL', Num, Smooth);
    [HammingLoss,RankingLoss,Coverage,Average_Precision,macrof1,microf1,Outputs,Pre_Labels]=...
        MLKNN_test(train_data(:,f), PL', test_data(:,f), test_target, Num, Prior, PriorN, Cond, CondN);
    fprintf('-- Evaluation\n');
    tt_eval = test_target; tt_eval(tt_eval==0) = -1;   
    tmpResult = EvaluationAll(Pre_Labels, Outputs, tt_eval);
    iterResult(:,order) = iterResult(:,order) + tmpResult;
end

else
    if num_feature <= 100
        d = ceil(0.4*num_feature);
    elseif num_feature <=500
        d = ceil(0.3*num_feature);
    elseif num_feature <= 1000
        d = ceil(0.2*num_feature);
    else
        d = ceil(0.1*num_feature);
    end
    order = 1;
    iterResult = zeros(15, 1);
    f = index(1:d);
    [Prior,PriorN,Cond,CondN]=MLKNN_train(train_data(:,f),train_data,Num,Smooth);
    [HammingLoss,RankingLoss,Coverage,Average_Precision,macrof1,microf1,Outputs,Pre_Labels]=...
        MLKNN_test(train_data(:,f), train_data, test_data(:,f),test_target,Num,Prior,PriorN,Cond,CondN);
    fprintf('-- Evaluation\n');
    tmpResult = EvaluationAll(Pre_Labels,Outputs,test_target);
    iterResult(:,order) = iterResult(:,order) + tmpResult;
end

Avg_Result      = zeros(15,2);
Avg_Result(:,1) = mean(iterResult,2);
Avg_Result(:,2) = std(iterResult,1,2);

figure; hold on;
plot(d_list, iterResult(12,:), 'DisplayName', model_name, 'Marker', 'h');
grid on; hold off;
xlabel('Selected Features (d)');  
ylabel('AP');
title(data_name);
legend('Location','best');
fprintf(' %10.4f    ', iterResult(14,:));

metric_names = { ...
    'Hamming Loss (lower better)', ...
    'Example-based Accuracy (higher better)', ...
    'Example-based Precision (higher better)', ...
    'Example-based Recall (higher better)', ...
    'Example-based F1-measure (higher better)', ...
    'Subset Accuracy (higher better)', ...
    'Label-based Accuracy (higher better)', ...
    'Label-based Precision (higher better)', ...
    'Label-based Recall (higher better)', ...
    'Label-based F1-measure (higher better)', ...
    'Micro-F1 Measure (higher better)', ...
    'Average Precision (higher better)', ...
    'One Error (lower better)', ...
    'Ranking Loss (lower better)', ...
    'Coverage (lower better)' ...
    };

disp('--- Final Evaluation Metrics ---');
disp('Metric Name                                   Mean         Std');
for i = 1:15
    fprintf('%-40s  %10.4f    %10.4f\n', metric_names{i}, Avg_Result(i,1), Avg_Result(i,2));
end

