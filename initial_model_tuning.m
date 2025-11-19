%% initial_model_tuning.m
% One-vs-all MLP with fixed 1:5 ratio and basic overfitting control
% Aligned with preprocess_csvs.m, run_benchmark.m, and improved initial_model.m

clear all; close all; clc;
rng(100);  % Reproducibility

%% Config
userRange_min = 1;
userRange_max = 10;
numUsers      = userRange_max - userRange_min + 1;

TrainTargetImposterRatio = 1/5;  % Genuine : Imposters = 1 : 5
dropoutRate              = 0.3;  % meta only (not applied directly in feedforwardnet)
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
        tField    = char(fieldnames(trainData));
        sField    = char(fieldnames(testData));
        userData(user).trainFeatures = trainData.(tField);
        userData(user).testFeatures  = testData.(sField);
        
        fprintf('User %d: train %dx%d, test %dx%d\n', ...
            user, size(userData(user).trainFeatures,1), size(userData(user).trainFeatures,2), ...
                  size(userData(user).testFeatures,1),  size(userData(user).testFeatures,2));
    else
        fprintf('Missing data for user %d\n', user);
        userData(user).trainFeatures = [];
        userData(user).testFeatures  = [];
    end
end

%% Storage
userMetrics        = zeros(numUsers, 14);
userPerformance    = zeros(numUsers, 3);               % [totalTime, memoryMB, throughput]
userSimilarityData = cell(3, numUsers, numUsers);      % means, mids, variations
models             = cell(numUsers, 1);

%% Train + evaluate per user
for targetUser = userRange_min:userRange_max
    fprintf('\n===== Training model for User %d =====\n', targetUser);

    % Skip if missing data
    if isempty(userData(targetUser).trainFeatures) || isempty(userData(targetUser).testFeatures)
        fprintf('Skipping user %d due to missing train/test data.\n', targetUser);
        continue;
    end
    
    % ---------- Build training set ----------
    XTrainTarget            = userData(targetUser).trainFeatures;
    trainTargetSampleCount  = size(XTrainTarget, 1);
    
    trainImposterSampleCount = round(trainTargetSampleCount * (1/TrainTargetImposterRatio));
    numOtherUsers            = numUsers - 1;
    trainSamplesPerImposter  = max(floor(trainImposterSampleCount / max(numOtherUsers,1)), 1);
    
    XTrain = XTrainTarget;
    yTrain = ones(trainTargetSampleCount, 1);   % genuine = 1
    
    impFeat = [];
    impLbl  = [];
    for u = userRange_min:userRange_max
        if u == targetUser || u == leaveOutUsersList(targetUser)
            continue;
        end
        if isempty(userData(u).trainFeatures)
            continue;
        end
        
        currFeat       = userData(u).trainFeatures;
        availableTrain = size(currFeat,1);
        samplesToTake  = min(trainSamplesPerImposter, availableTrain);
        if samplesToTake < 1, continue; end
        
        idx = randperm(availableTrain, samplesToTake);
        impFeat = [impFeat; currFeat(idx, :)];
        impLbl  = [impLbl;  zeros(samplesToTake, 1)];
    end
    
    XTrain = [XTrain; impFeat];
    yTrain = [yTrain; impLbl];
    
    % Ensure both classes exist in training
    if numel(unique(yTrain)) < 2
        fprintf('User %d: training set has only one class. Skipping.\n', targetUser);
        continue;
    end
    
    % ---------- Build testing set ----------
    XTestTarget           = userData(targetUser).testFeatures;
    testTargetSampleCount = size(XTestTarget, 1);
    
    XTest = XTestTarget;
    yTest = ones(testTargetSampleCount, 1);
    testUserLabels = ones(testTargetSampleCount, 1)*targetUser;
    
    impFeatTest = [];
    impLblTest  = [];
    
    % Make testing imposter load comparable to target load
    testImposterSampleCount = testTargetSampleCount * (numUsers - 1);
    testSamplesPerImposter  = max(floor(testImposterSampleCount / max(numUsers-1,1)), 1);
    
    for u = userRange_min:userRange_max
        if u == targetUser, continue; end
        if isempty(userData(u).testFeatures)
            continue;
        end
        
        currFeat        = userData(u).testFeatures;
        availableTest   = size(currFeat,1);
        samplesToTake   = min(testSamplesPerImposter, availableTest);
        if samplesToTake < 1, continue; end
        
        idx             = randperm(availableTest, samplesToTake);
        impFeatTest     = [impFeatTest; currFeat(idx,:)];
        impLblTest      = [impLblTest;  zeros(samplesToTake,1)];
        testUserLabels  = [testUserLabels; u*ones(samplesToTake,1)];
    end
    
    XTest = [XTest; impFeatTest];
    yTest = [yTest; impLblTest];
    
    % Ensure test set has both classes
    if numel(unique(yTest)) < 2
        fprintf('User %d: test set has only one class. Skipping.\n', targetUser);
        continue;
    end
    
    % ---------- Shuffle training ----------
    nTrain = size(XTrain,1);
    idxSh  = randperm(nTrain);
    XTrain = XTrain(idxSh,:);
    yTrain = yTrain(idxSh,:);
    
    % ---------- Network definition ----------
    net = feedforwardnet(131, 'trainscg');
    net.userdata.note                     = 'Initial FFNN w/ leave-out imposters (tuning)';
    net.userdata.trainTargetImposterRatio = sprintf('1:%d', round(1/TrainTargetImposterRatio));
    net.userdata.dropoutRate              = dropoutRate;
    net.userdata.l2RegParam               = l2RegParam;
    net.userdata.performanceGoal          = performanceGoal;
    net.userdata.minGrad                  = minGrad;
    net.userdata.earlyStoppingPatience    = earlyStoppingPatience;
    net.userdata.maxEpochs                = maxEpochs;
    net.userdata.learningRate             = learningRate;
    net.userdata.targetUser               = sprintf('User %d', targetUser);
    
    net.performFcn              = 'crossentropy';
    net.layers{1}.transferFcn   = 'tansig';
    net.layers{end}.transferFcn = 'logsig';   % output in [0,1] for crossentropy
    
    net.trainParam.epochs   = maxEpochs;
    net.trainParam.goal     = performanceGoal;
    net.trainParam.min_grad = minGrad;
    net.performParam.regularization = l2RegParam;
    net.trainParam.max_fail = earlyStoppingPatience;
    net.trainParam.lr       = learningRate;
    
    % Input processing (standard + mapminmax)
    net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
    % Use default output processing (no invalid net.outputs{2})
    
    % ---------- Training ----------
    tic;
    [net, tr] = train(net, XTrain', yTrain');
    trainTime = toc;
    
    models{targetUser} = net;
    
    % Memory usage
    info  = whos('net');
    memMB = info.bytes / (1024^2);
    
    % ---------- Inference ----------
    tic;
    yScores = net(XTest')';   % continuous outputs [0,1]
    inferenceTime = toc;
    throughput    = size(XTest,1) / max(inferenceTime, eps);
    
    userPerformance(targetUser,:) = [trainTime+inferenceTime, memMB, throughput];
    
    % Store similarities (scores per true user label)
    modelUserSimilarities = [testUserLabels, yScores];
    
    % Binarize at 0.5 for confusion-based metrics
    yPred = double(yScores > 0.5);
    
    % ---------- Metrics ----------
    epsVal = 1e-12;
    
    tp = sum(yPred==1 & yTest==1);
    tn = sum(yPred==0 & yTest==0);
    fp = sum(yPred==1 & yTest==0);
    fn = sum(yPred==0 & yTest==1);
    
    precision   = tp / max(tp + fp, epsVal);
    recall      = tp / max(tp + fn, epsVal);
    specificity = tn / max(tn + fp, epsVal);
    accuracy    = (tp + tn) / max(tp + tn + fp + fn, epsVal);
    f1_score    = 2 * precision * recall / max(precision + recall, epsVal);
    
    % FAR & FRR from confusion (single operating point)
    fpr  = fp / max(fp + tn, epsVal);   % FAR
    fnr  = fn / max(fn + tp, epsVal);   % FRR
    eer  = (fpr + fnr) / 2;
    
    % MCC
    denomMCC = sqrt( max(tp+fp,epsVal) * max(tp+fn,epsVal) * ...
                     max(tn+fp,epsVal) * max(tn+fn,epsVal) );
    mcc = ((tp*tn) - (fp*fn)) / max(denomMCC, epsVal);
    
    % AUC & FAR/FRR curve from continuous scores
    [FPR_curve, TPR_curve, Troc, AUC] = perfcurve(yTest, yScores, 1); %#ok<ASGLU>
    FAR_curve = FPR_curve;
    FRR_curve = 1 - TPR_curve;
    [~, eerIdxCurve] = min(abs(FAR_curve - FRR_curve));
    EER_curve = (FAR_curve(eerIdxCurve) + FRR_curve(eerIdxCurve)) / 2;
    
    % ---------- Similarity statistics ----------
    sim_means = zeros(1, numUsers);
    sim_mids  = zeros(1, numUsers);
    sim_vars  = zeros(1, numUsers);
    
    for u = userRange_min:userRange_max
        idxu = find(modelUserSimilarities(:,1) == u);
        vals = modelUserSimilarities(idxu,2);
        if isempty(vals)
            sim_means(u) = NaN;
            sim_mids(u)  = NaN;
            sim_vars(u)  = NaN;
            continue;
        end
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
        f1_score, mcc, fpr*100, fnr*100, EER_curve*100, AUC, ...
        size(XTrain,1), trainTargetSampleCount, sum(impLbl), ...
        size(XTest,1) ...
    ];
    
    % ---------- Print ----------
    fprintf('Accuracy:    %.2f%%\n', accuracy*100);
    fprintf('Precision:   %.2f%%\n', precision*100);
    fprintf('Recall:      %.2f%%\n', recall*100);
    fprintf('Specificity: %.2f%%\n', specificity*100);
    fprintf('F1-score:    %.2f%%\n', f1_score*100);
    fprintf('MCC:         %.4f\n',  mcc);
    fprintf('FAR (conf):  %.2f%%\n', fpr*100);
    fprintf('FRR (conf):  %.2f%%\n', fnr*100);
    fprintf('EER (curve): %.2f%%\n', EER_curve*100);
    fprintf('AUC:         %.4f\n',  AUC);
    fprintf('Time:        %.4fs, Mem: %.2fMB, Throughput: %.2f samples/s\n', ...
        trainTime+inferenceTime, memMB, throughput);
    
    % ---------- Confusion matrix ----------
    figure;
    plotconfusion(yTest', yPred');
    title(sprintf('Confusion Matrix - User %d', targetUser));
    
    % ---------- ROC ----------
    figure;
    plot(FPR_curve, TPR_curve); hold on;
    plot(FAR_curve(eerIdxCurve), TPR_curve(eerIdxCurve), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
    text(FAR_curve(eerIdxCurve), TPR_curve(eerIdxCurve), ...
        sprintf('  EER = %.2f%%', EER_curve*100), 'VerticalAlignment','bottom');
    xlabel('False Acceptance Rate (FAR)');
    ylabel('True Positive Rate (TPR)');
    title(sprintf('ROC - User %d (AUC = %.3f)', targetUser, AUC));
    grid on; hold off;
end

%% Averages / summary
validUsers     = find(sum(userMetrics,2)~=0);
avgMetrics     = mean(userMetrics(validUsers,:), 1);
avgPerformance = mean(userPerformance(validUsers,:), 1);

results = struct( ...
    'Ratio',                   '1:5', ...
    'AvgAccuracy',             avgMetrics(1)*100, ...
    'AvgPrecision',            avgMetrics(2)*100, ...
    'AvgRecall',               avgMetrics(3)*100, ...
    'AvgSpecificity',          avgMetrics(4)*100, ...
    'AvgF1Score',              avgMetrics(5)*100, ...
    'AvgMCC',                  avgMetrics(6), ...
    'AvgFAR',                  avgMetrics(7), ...
    'AvgFRR',                  avgMetrics(8), ...
    'AvgEER',                  avgMetrics(9), ...
    'AvgAUC',                  avgMetrics(10), ...
    'AvgTrainingSetSize',      avgMetrics(11), ...
    'AvgTrainTargetSamples',   avgMetrics(12), ...
    'AvgTrainImposterSamples', avgMetrics(13), ...
    'AvgTestSetSize',          avgMetrics(14), ...
    'AvgTotalTime',            avgPerformance(1), ...
    'AvgMemoryUsage',          avgPerformance(2), ...
    'AvgThroughput',           avgPerformance(3) ...
);

fprintf('\n==== Network summary (tuning model) ====\n');
% Use first valid user for input dim
validUserForDim = validUsers(1);
fprintf('Input dim: %d\n', size(userData(validUserForDim).trainFeatures,2));
fprintf('Hidden layer: 131 (tansig)\n');
fprintf('Output: 1 (logsig)\n');
fprintf('TrainFcn: trainscg, PerformFcn: crossentropy\n');

fprintf('\n==== Performance Benchmarks ====\n');
fprintf('Avg Total Time:  %.4fs (±%.4f)\n', mean(userPerformance(validUsers,1)), std(userPerformance(validUsers,1)));
fprintf('Avg Mem Usage:   %.2fMB (±%.2f)\n', mean(userPerformance(validUsers,2)), std(userPerformance(validUsers,2)));
fprintf('Avg Throughput:  %.2f (±%.2f) samples/s\n', mean(userPerformance(validUsers,3)), std(userPerformance(validUsers,3)));

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
    mean(userPerformance(validUsers,1)), ...
    mean(userPerformance(validUsers,2)), ...
    mean(userPerformance(validUsers,3)), ...
    mean(userMetrics(validUsers,1)*100), ...
    mean(userMetrics(validUsers,2)*100), ...
    mean(userMetrics(validUsers,3)*100), ...
    mean(userMetrics(validUsers,4)*100), ...
    mean(userMetrics(validUsers,5)*100), ...
    mean(userMetrics(validUsers,6)), ...
    mean(userMetrics(validUsers,7)), ...
    mean(userMetrics(validUsers,8)), ...
    mean(userMetrics(validUsers,9)), ...
    mean(userMetrics(validUsers,10)), ...
    'VariableNames', { ...
        'Avg_TotalTime_sec', 'Avg_MemoryUsage_MB', 'Avg_Throughput_samples_per_sec', ...
        'Avg_Accuracy', 'Avg_Precision', 'Avg_Recall', 'Avg_Specificity', 'Avg_F1_Score', ...
        'Avg_MCC', 'Avg_FAR', 'Avg_FRR', 'Avg_EER', 'Avg_AUC' ...
    });

fprintf('\n==== Summary Table ====\n');
disp(summaryTable);
disp('Overall Metrics:'); disp(overallMetrics);

%% Similarity heatmap
similarityMatrix = zeros(numUsers, numUsers);
labelStrings     = cell(numUsers, numUsers);

for i = 1:numUsers
    for j = 1:numUsers
        val = cell2mat(userSimilarityData(1,i,j));
        mid = cell2mat(userSimilarityData(2,i,j));
        var = cell2mat(userSimilarityData(3,i,j));
        similarityMatrix(i,j) = val;
        if isnan(val)
            labelStrings{i,j} = 'NaN';
        else
            labelStrings{i,j} = sprintf('%.2f\nM: %.2f\n(±%.3f)', val, mid, var);
        end
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
title('User similarity scores for each user model (tuning)');

save('benchmark_results_initial_model_tuning.mat', 'summaryTable','overallMetrics','results');
save('user_authentication_models_initial_tuning.mat', 'models','userSimilarityData');
