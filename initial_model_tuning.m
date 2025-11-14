%% initial_model_tuning.m
% One-vs-all MLP with fixed 1:5 ratio and basic overfitting control

clear all; close all; clc;
rng(100);  % Reproducibility

%% Config
userRange_min = 1;
userRange_max = 10;
numUsers      = userRange_max - userRange_min + 1;

TrainTargetImposterRatio = 1/5;  % Genuine : Imposters = 1 : 5
dropoutRate              = 0.3;
l2RegParam               = 1e-4;
performanceGoal          = 1e-5;
minGrad                  = 1e-6;
earlyStoppingPatience    = 10;
maxEpochs                = 500;
learningRate             = 0.01;

filePatternsTrain = 'Acc_TimeD_FreqD_FDay';
filePatternsTest  = 'Acc_TimeD_FreqD_MDay';
dataFolder        = 'dataset';

% Fixed leave-out list (one impostor user not seen during training)
leaveOutUsersList = [6, 3, 2, 5, 6, 1, 9, 7, 7, 3];

%% Load data
fprintf('Loading data for each user...\n');

userData = struct('trainFeatures', [], 'testFeatures', []);
userData = repmat(userData, 1, userRange_max);

for user = userRange_min:userRange_max
    userStr   = sprintf('U%02d', user);
    trainFile = fullfile(dataFolder, [userStr '_' filePatternsTrain '.mat']);
    testFile  = fullfile(dataFolder, [userStr '_' filePatternsTest  '.mat']);
    
    if exist(trainFile, 'file') && exist(testFile, 'file')
        trainData = load(trainFile);
        testData  = load(testFile);
        userData(user).trainFeatures = trainData.(char(fieldnames(trainData)));
        userData(user).testFeatures  = testData.(char(fieldnames(testData)));
        
        fprintf('User %d: train %dx%d, test %dx%d\n', ...
            user, size(userData(user).trainFeatures,1), size(userData(user).trainFeatures,2), ...
                  size(userData(user).testFeatures,1),  size(userData(user).testFeatures,2));
    else
        fprintf('Missing data for user %d\n', user);
    end
end

%% Storage
userMetrics     = zeros(numUsers, 14);
userPerformance = zeros(numUsers, 3);               % [time, memoryMB, throughput]
userSimilarityData = cell(3, numUsers, numUsers);   % means, mids, variations

%% Train + evaluate per user
for targetUser = userRange_min:userRange_max
    fprintf('\n===== Training model for User %d =====\n', targetUser);
    
    % ---------- Build training set ----------
    trainTargetSampleCount    = size(userData(targetUser).trainFeatures, 1);
    trainImposterSampleCount  = trainTargetSampleCount * (1/TrainTargetImposterRatio);
    trainSamplesPerImposter   = floor(trainImposterSampleCount / (numUsers - 1));
    
    XTrain = userData(targetUser).trainFeatures;
    yTrain = ones(trainTargetSampleCount, 1);
    
    impFeat = [];
    impLbl  = [];
    for u = userRange_min:userRange_max
        if u == targetUser || u == leaveOutUsersList(targetUser)
            continue;
        end
        currFeat = userData(u).trainFeatures;
        idx = randperm(size(currFeat,1), min(trainSamplesPerImposter, size(currFeat,1)));
        impFeat = [impFeat; currFeat(idx, :)];
        impLbl  = [impLbl;  zeros(length(idx), 1)];
    end
    
    XTrain = [XTrain; impFeat];
    yTrain = [yTrain; impLbl];
    
    % Expected imposters ~ (numUsers-2)*trainSamplesPerImposter
    assert(sum(yTrain==1) == trainTargetSampleCount);
    
    % ---------- Build testing set ----------
    testTargetSampleCount   = size(userData(targetUser).testFeatures, 1);
    testImposterSampleCount = 324;
    testSamplesPerImposter  = floor(testImposterSampleCount / (numUsers - 1));
    
    XTest = userData(targetUser).testFeatures;
    yTest = ones(testTargetSampleCount, 1);
    testUserLabels = ones(testTargetSampleCount, 1)*targetUser;
    
    impFeatTest = [];
    impLblTest  = [];
    
    for u = userRange_min:userRange_max
        if u == targetUser, continue; end
        currFeat = userData(u).testFeatures;
        idx = randperm(size(currFeat,1), min(testSamplesPerImposter, size(currFeat,1)));
        impFeatTest = [impFeatTest; currFeat(idx,:)];
        impLblTest  = [impLblTest;  zeros(length(idx),1)];
        testUserLabels = [testUserLabels; u*ones(length(idx),1)];
    end
    
    XTest = [XTest; impFeatTest];
    yTest = [yTest; impLblTest];
    
    assert(sum(yTest==1) == testTargetSampleCount);
    
    % ---------- Network definition ----------
    net = feedforwardnet(131, 'trainscg');
    net.userdata.note                     = 'Initial FFNN w/ leave-out imposters';
    net.userdata.trainTargetImposterRatio = sprintf('1:%d', round(1/TrainTargetImposterRatio));
    net.userdata.dropoutRate              = dropoutRate;
    net.userdata.l2RegParam               = l2RegParam;
    net.userdata.performanceGoal          = performanceGoal;
    net.userdata.minGrad                  = minGrad;
    net.userdata.earlyStoppingPatience    = earlyStoppingPatience;
    net.userdata.maxEpochs                = maxEpochs;
    net.userdata.learningRate             = learningRate;
    net.userdata.targetUser               = sprintf('User %d', targetUser);
    
    net.performFcn = 'crossentropy';
    net.layers{1}.transferFcn  = 'logsig';
    net.layers{end}.transferFcn = 'tansig';
    
    net.trainParam.epochs   = maxEpochs;
    net.trainParam.goal     = performanceGoal;
    net.trainParam.min_grad = minGrad;
    net.performParam.regularization = l2RegParam;
    net.trainParam.max_fail = earlyStoppingPatience;
    net.trainParam.lr       = learningRate;
    
    net.inputs{1}.processFcns  = {'removeconstantrows','mapstd','processpca'};
    net.outputs{2}.processFcns = {'mapminmax'};
    
    % ---------- Training ----------
    tic;
    [net, tr] = train(net, XTrain', yTrain');
    trainTime = toc;
    
    models{targetUser} = net; %#ok<SAGROW>
    
    % Memory usage
    info = whos('net');
    memMB = info.bytes / (1024^2);
    
    % ---------- Inference ----------
    tic;
    yScores = net(XTest')';   % continuous outputs
    inferenceTime = toc;
    throughput = size(XTest,1) / inferenceTime;
    
    userPerformance(targetUser,:) = [trainTime+inferenceTime, memMB, throughput];
    
    % Store similarities (scores per true user label)
    modelUserSimilarities = [testUserLabels, yScores];
    
    % Binarize at 0.5 for confusion-based metrics
    yPred = double(yScores > 0.5);
    
    % ---------- Metrics ----------
    tp = sum(yPred==1 & yTest==1);
    tn = sum(yPred==0 & yTest==0);
    fp = sum(yPred==1 & yTest==0);
    fn = sum(yPred==0 & yTest==1);
    
    precision   = tp / (tp + fp + eps);
    recall      = tp / (tp + fn + eps);
    specificity = tn / (tn + fp + eps);
    accuracy    = (tp + tn) / (tp + tn + fp + fn + eps);
    f1_score    = 2 * precision * recall / (precision + recall + eps);
    fpr         = fp / (fp + tn + eps);
    fnr         = fn / (fn + tp + eps);
    eer         = (fpr + fnr) / 2;
    mcc         = ((tp*tn) - (fp*fn)) / sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) + eps);
    
    % AUC from continuous scores
    [Xroc, Yroc, Troc, AUC] = perfcurve(yTest, yScores, 1); %#ok<ASGLU>
    
    % ---------- Similarity statistics ----------
    sim_means = zeros(1, numUsers);
    sim_mids  = zeros(1, numUsers);
    sim_vars  = zeros(1, numUsers);
    
    for u = userRange_min:userRange_max
        idxu = find(modelUserSimilarities(:,1) == u);
        vals = modelUserSimilarities(idxu,2);
        sim_means(u) = mean(vals);
        vmin         = min(vals);
        vmax         = max(vals);
        mid          = (vmin + vmax)/2;
        sim_mids(u)  = mid;
        sim_vars(u)  = vmax - mid;
    end
    
    userSimilarityData(1, targetUser, :) = num2cell(sim_means);
    userSimilarityData(2, targetUser, :) = num2cell(sim_mids);
    userSimilarityData(3, targetUser, :) = num2cell(sim_vars);
    
    % ---------- Store metrics ----------
    userMetrics(targetUser,:) = [ ...
        accuracy, precision, recall, specificity, ...
        f1_score, mcc, fpr*100, fnr*100, eer*100, AUC, ...
        size(XTrain,1), trainTargetSampleCount, trainImposterSampleCount, ...
        size(XTest,1) ...
    ];
    
    % ---------- Print ----------
    fprintf('Accuracy:   %.2f%%\n', accuracy*100);
    fprintf('Precision:  %.2f%%\n', precision*100);
    fprintf('Recall:     %.2f%%\n', recall*100);
    fprintf('Specificity:%.2f%%\n', specificity*100);
    fprintf('F1-score:   %.2f%%\n', f1_score*100);
    fprintf('MCC:        %.4f\n',  mcc);
    fprintf('FAR:        %.2f%%\n', fpr*100);
    fprintf('FRR:        %.2f%%\n', fnr*100);
    fprintf('EER:        %.2f%%\n', eer*100);
    fprintf('AUC:        %.4f\n',  AUC);
    fprintf('Time:       %.4fs, Mem: %.2fMB, Throughput: %.2f samples/s\n', ...
        trainTime+inferenceTime, memMB, throughput);
    
    % ---------- Confusion matrix ----------
    figure;
    plotconfusion(yTest', yPred');
    title(sprintf('Confusion Matrix - User %d', targetUser));
    
    % ---------- ROC ----------
    figure;
    plot(Xroc, Yroc);
    xlabel('False Positive Rate'); ylabel('True Positive Rate');
    title(sprintf('ROC - User %d (AUC = %.3f)', targetUser, AUC));
    grid on;
end

%% Averages / summary
avgMetrics     = mean(userMetrics, 1);
avgPerformance = mean(userPerformance, 1);

results = struct( ...
    'Ratio',                 '1:5', ...
    'AvgAccuracy',           avgMetrics(1)*100, ...
    'AvgPrecision',          avgMetrics(2)*100, ...
    'AvgRecall',             avgMetrics(3)*100, ...
    'AvgSpecificity',        avgMetrics(4)*100, ...
    'AvgF1Score',            avgMetrics(5)*100, ...
    'AvgMCC',                avgMetrics(6), ...
    'AvgFAR',                avgMetrics(7), ...
    'AvgFRR',                avgMetrics(8), ...
    'AvgEER',                avgMetrics(9), ...
    'AvgAUC',                avgMetrics(10), ...
    'AvgTrainingSetSize',    avgMetrics(11), ...
    'AvgTrainTargetSamples', avgMetrics(12), ...
    'AvgTrainImposterSamples', avgMetrics(13), ...
    'AvgTestSetSize',        avgMetrics(14), ...
    'AvgTotalTime',          avgPerformance(1), ...
    'AvgMemoryUsage',        avgPerformance(2), ...
    'AvgThroughput',         avgPerformance(3) ...
);

fprintf('\n==== Network summary ====\n');
fprintf('Input dim: %d\n', size(userData(1).trainFeatures,2));
fprintf('Hidden layer: 131 (logsig)\n');
fprintf('Output: 1 (tansig)\n');
fprintf('TrainFcn: trainscg, PerformFcn: crossentropy\n');

fprintf('\n==== Performance Benchmarks ====\n');
fprintf('Avg Time:  %.4fs (±%.4f)\n', mean(userPerformance(:,1)), std(userPerformance(:,1)));
fprintf('Avg Mem:   %.2fMB (±%.2f)\n', mean(userPerformance(:,2)), std(userPerformance(:,2)));
fprintf('Avg Thput: %.2f (±%.2f) samples/s\n', mean(userPerformance(:,3)), std(userPerformance(:,3)));

summaryTable = table((1:numUsers)', ...
    userPerformance(:,1), userPerformance(:,2), userPerformance(:,3), ...
    userMetrics(:,1)*100, userMetrics(:,2)*100, userMetrics(:,3)*100, ...
    userMetrics(:,4)*100, userMetrics(:,5)*100, userMetrics(:,6), ...
    userMetrics(:,7), userMetrics(:,8), userMetrics(:,9), userMetrics(:,10), ...
    'VariableNames', { ...
        'User', 'TotalTime_sec', 'MemoryUsage_MB', 'Throughput_samples_per_sec', ...
        'Accuracy', 'Precision', 'Recall', 'Specificity', 'F1_Score', ...
        'MCC', 'FAR', 'FRR', 'EER', 'AUC' ...
    });

overallMetrics = table( ...
    mean(userPerformance(:,1)), ...
    mean(userPerformance(:,2)), ...
    mean(userPerformance(:,3)), ...
    mean(userMetrics(:,1)*100), ...
    mean(userMetrics(:,2)*100), ...
    mean(userMetrics(:,3)*100), ...
    mean(userMetrics(:,4)*100), ...
    mean(userMetrics(:,5)*100), ...
    mean(userMetrics(:,6)), ...
    mean(userMetrics(:,7)), ...
    mean(userMetrics(:,8)), ...
    mean(userMetrics(:,9)), ...
    mean(userMetrics(:,10)), ...
    'VariableNames', { ...
        'Avg_TotalTime_sec', 'Avg_MemoryUsage_MB', 'Avg_Throughput_samples_per_sec', ...
        'Avg_Accuracy', 'Avg_Precision', 'Avg_Recall', 'Avg_Specificity', 'Avg_F1_Score', ...
        'Avg_MCC', 'Avg_FAR', 'Avg_FRR', 'Avg_EER', 'Avg_AUC' ...
    });

fprintf('\n==== Summary Table ====\n');
disp(summaryTable);
disp('Overall Metrics:'); disp(overallMetrics);

% Similarity heatmap
similarityMatrix = zeros(numUsers, numUsers);
labelStrings     = cell(numUsers, numUsers);

for i = 1:numUsers
    for j = 1:numUsers
        val = cell2mat(userSimilarityData(1,i,j));
        mid = cell2mat(userSimilarityData(2,i,j));
        var = cell2mat(userSimilarityData(3,i,j));
        similarityMatrix(i,j) = val;
        labelStrings{i,j} = sprintf('%.2f\nM: %.2f\n(±%.3f)', val, mid, var);
    end
end

figure('Position',[100 100 800 600]);
imagesc(similarityMatrix);
colormap(parula);
c = colorbar; c.Label.String = 'Similarity Score';
axis square;

for i = 1:numUsers
    for j = 1:numUsers
        text(j, i, labelStrings{i,j}, ...
            'HorizontalAlignment','center', 'Color','k', 'FontSize',8);
    end
end
set(gca,'XTick',1:numUsers,'XTickLabel',userRange_min:userRange_max);
set(gca,'YTick',1:numUsers,'YTickLabel',userRange_min:userRange_max);
xlabel("User N's similarity score");
ylabel("User N's model");
title('User similarity scores for each user model');

save('benchmark_results_initial_model.mat', 'summaryTable','overallMetrics','results');
save('user_authentication_models_initial.mat', 'models');
