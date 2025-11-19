%% optimization_SVM_no_louo_of_model_fast.m  (FAST MODE)
% Enhanced User Authentication System with Optimized Neural Network
% FAST VERSION: lighter GA, fewer NN trials, fewer epochs, limited plotting.
% This version uses ALL other users as imposters (no leave-out user).

clear all; close all; clc;
rng(100);  % For reproducibility

%% ===================== CONFIGURATION ====================================

% User range
userRange_min = 1;
userRange_max = 10;
numUsers      = userRange_max - userRange_min + 1;

% Core ratio
TrainTargetImposterRatio = 1/5;   % Fixed ratio 1:5

% NN basic hyperparameters (can still be tuned)
performanceGoal        = 1e-5;
minGrad                = 1e-6;
earlyStoppingPatience  = 10;
maxEpochs              = 100;      % REDUCED from 300 for speed
learningRateDefault    = 0.005;
batchSize              = 32;

% NN architecture (from original no-LOU script)
trainFcn                    = 'trainscg';
hiddenLayerSizes            = [57 28];
hiddenLayerActivationFcns   = {'logsig'; 'logsig'};
outputLayerActivationFcn    = 'tansig';
performanceFcn              = 'crossentropy';

% Files
filePatternsTrain = 'Acc_TimeD_FreqD_FDay';
filePatternsTest  = 'Acc_TimeD_FreqD_MDay';

% FAST MODE switches
MAX_FEATURES_FOR_GA     = 40;   % only first K features used by GA (for speed)
GA_PopulationSize       = 10;   % reduced
GA_MaxGenerations       = 5;    % reduced
RANDOM_SEARCH_ITER      = 3;    % reduced random search loops per user
PLOT_ROC_AND_CONFUSION  = false; % disable heavy plotting in fast mode

%% ===================== DATA LOADING =====================================

fprintf('Loading data for each user...\n');

% Initialize storage datasets
userData = struct('trainFeatures', [], 'testFeatures', []);
userData = repmat(userData, 1, userRange_max);

for user = userRange_min:userRange_max
    userStr   = sprintf('U%02d', user);
    trainFile = fullfile('dataset', [userStr '_' filePatternsTrain '.mat']);
    testFile  = fullfile('dataset', [userStr '_' filePatternsTest  '.mat']);

    if exist(trainFile, 'file') && exist(testFile, 'file')
        trainData = load(trainFile);
        testData  = load(testFile);

        userData(user).trainFeatures = trainData.(char(fieldnames(trainData)));
        userData(user).testFeatures  = testData.(char(fieldnames(testData)));

        [r,c] = size(userData(user).trainFeatures);
        fprintf('User %d: %d train samples x %d features\n', user, r, c);
    else
        fprintf('Missing data files for user %d\n', user);
    end
end

%% ===================== FEATURE SELECTION + TRAINING =====================

userMetrics          = zeros(numUsers, 14);
userPerformance      = zeros(numUsers, 3);   % [totalTime, memoryMB, throughput]
userSimilarityData   = cell(3, numUsers, numUsers);
selectedFeaturesPerUser = cell(numUsers, 1);

fprintf('\nTraining neural network models (FAST, no LOU)...\n');

for targetUser = userRange_min:userRange_max
    fprintf('\n=== Training model for User %d ===\n', targetUser);
    fprintf('-----------------------------------------\n');

    %% Step 1: GA-based feature selection (FAST, NO-LOU)
    [selectedFeatures, featureSelectionTime] = ...
        performGeneticFeatureSelection_FAST_noLOU(userData, targetUser, ...
                                                  userRange_min, userRange_max, ...
                                                  MAX_FEATURES_FOR_GA, ...
                                                  GA_PopulationSize, ...
                                                  GA_MaxGenerations);

    selectedFeaturesPerUser{targetUser} = selectedFeatures;
    fprintf('  GA feature selection finished in %.2f seconds\n', featureSelectionTime);
    fprintf('  Selected %d features\n', numel(selectedFeatures));

    % Initialize performance with feature selection time
    userPerformance(targetUser, :) = [featureSelectionTime, 0, 0];

    %% Step 2: Build training data (target vs imposters) using selected features
    XTrain_Target = userData(targetUser).trainFeatures(:, selectedFeatures);
    trainTargetSampleCount   = size(XTrain_Target, 1);
    trainImposterSampleCount = trainTargetSampleCount * (1/TrainTargetImposterRatio);
    trainSamplesPerImposter  = floor(trainImposterSampleCount/(numUsers-1));

    XTrain = XTrain_Target;
    yTrain = ones(trainTargetSampleCount, 1);

    trainImposterFeatures = [];
    trainImposterLabels   = [];

    for imposterUser = userRange_min:userRange_max
        if imposterUser ~= targetUser
            available = size(userData(imposterUser).trainFeatures,1);
            if available == 0, continue; end

            k   = min(trainSamplesPerImposter, available);
            idx = randperm(available, k);

            trainImposterFeatures = [trainImposterFeatures;
                userData(imposterUser).trainFeatures(idx, selectedFeatures)];
            trainImposterLabels = [trainImposterLabels;
                zeros(k, 1)];
        end
    end

    XTrain = [XTrain; trainImposterFeatures];
    yTrain = [yTrain; trainImposterLabels];

    fprintf('  Train positives: %d, negatives: %d\n', ...
        sum(yTrain==1), sum(yTrain==0));

    % Normalize (z-score)
    [XTrain, mu, sigma] = zscore(XTrain);

    %% Step 3: Random search hyperparameter tuning (FAST)
    fprintf('  Starting random search (%d iterations)...\n', RANDOM_SEARCH_ITER);
    bestAccuracy = 0;
    bestNet      = [];
    tic;
    for it = 1:RANDOM_SEARCH_ITER
        % Random hyperparams (compact range)
        lr = 10^(-3 + rand*1.5);   % ~0.001 to 0.03
        l2 = 10^(-4 + rand*1.5);   % ~1e-4 to 3e-3

        % Define neural network
        net = feedforwardnet(hiddenLayerSizes, trainFcn);
        net.userdata.note                     = "FAST NN (no LOU) with GA features";
        net.userdata.trainTargetImposterRatio = sprintf("1:%d", round(1/TrainTargetImposterRatio));
        net.userdata.performanceGoal          = performanceGoal;
        net.userdata.minGrad                  = minGrad;
        net.userdata.earlyStoppingPatience    = earlyStoppingPatience;
        net.userdata.maxEpochs                = maxEpochs;
        net.userdata.learningRate             = lr;
        net.userdata.batchSize                = batchSize;
        net.userdata.noOfFeatures             = size(XTrain, 2);
        net.userdata.targetUser               = sprintf('User %d', targetUser);

        % Layers
        for layerNo = 1:length(hiddenLayerActivationFcns)
            net.layers{layerNo}.transferFcn = hiddenLayerActivationFcns{layerNo};
        end
        net.layers{end}.transferFcn = outputLayerActivationFcn;
        net.performFcn              = performanceFcn;
        net.performParam.regularization = l2;

        % Training parameters
        net.trainParam.epochs   = maxEpochs;
        net.trainParam.goal     = performanceGoal;
        net.trainParam.min_grad = minGrad;
        net.trainParam.max_fail = earlyStoppingPatience;
        net.trainParam.lr       = lr;

        % Split
        net.divideParam.trainRatio = 0.7;
        net.divideParam.valRatio   = 0.15;
        net.divideParam.testRatio  = 0.15;

        % Disable GUI
        net.trainParam.showWindow      = false;
        net.trainParam.showCommandLine = false;

        % Train
        [net, tr] = train(net, XTrain', yTrain');

        % Validation accuracy
        yValPred = net(XTrain(tr.valInd,:)')';
        yValBin  = yValPred > 0.5;
        valAccuracy = sum(yValBin == yTrain(tr.valInd)) / numel(tr.valInd);

        fprintf('    Iter %d: lr=%.4f, l2=%.3e, valAcc=%.4f\n', ...
            it, lr, l2, valAccuracy);

        if valAccuracy > bestAccuracy
            bestAccuracy = valAccuracy;
            bestNet      = net;
        end
    end
    trainTimeRandomSearch = toc;
    fprintf('  Random search total time: %.2f sec (best valAcc=%.4f)\n', ...
        trainTimeRandomSearch, bestAccuracy);

    % Save best model & norm params
    models{targetUser} = bestNet;
    normalizationParams{targetUser} = struct('mu', mu, 'sigma', sigma);

    % Memory usage
    modelInfo    = whos('bestNet');
    memoryUsageMB = modelInfo.bytes / (1024^2);

    % Update performance timing
    userPerformance(targetUser, 1) = userPerformance(targetUser, 1) + trainTimeRandomSearch;
    userPerformance(targetUser, 2) = memoryUsageMB;
end

%% ===================== TESTING ==========================================

fprintf('\nTesting models (no LOU)...\n');

for targetUser = userRange_min:userRange_max
    fprintf('\n=== Testing model for User %d ===\n', targetUser);
    fprintf('-----------------------------------------\n');

    selectedFeatures = selectedFeaturesPerUser{targetUser};
    if isempty(selectedFeatures)
        warning('No selected features for user %d – skipping', targetUser);
        continue;
    end

    % Target test samples
    XTest_Target = userData(targetUser).testFeatures(:, selectedFeatures);
    testTargetSampleCount = size(XTest_Target, 1);

    % Imposter test samples (all other users)
    testImposterSampleCount = testTargetSampleCount * (numUsers-1);
    testSamplesPerImposter  = floor(testImposterSampleCount/(numUsers-1));

    XTest         = XTest_Target;
    yTest         = ones(testTargetSampleCount, 1);
    testUserLabels = ones(testTargetSampleCount, 1) * targetUser;

    testImposterFeatures = [];
    testImposterLabels   = [];

    for imposterUser = userRange_min:userRange_max
        if imposterUser ~= targetUser
            available = size(userData(imposterUser).testFeatures,1);
            if available == 0, continue; end

            k   = min(testSamplesPerImposter, available);
            idx = randperm(available, k);

            testImposterFeatures = [testImposterFeatures;
                userData(imposterUser).testFeatures(idx, selectedFeatures)];
            testImposterLabels = [testImposterLabels;
                zeros(k, 1)];
            testUserLabels = [testUserLabels;
                ones(k,1)*imposterUser];
        end
    end

    XTest = [XTest; testImposterFeatures];
    yTest = [yTest; testImposterLabels];

    fprintf('  Test positives: %d, negatives: %d\n', ...
        sum(yTest==1), sum(yTest==0));

    % Normalize with training stats
    mu    = normalizationParams{targetUser}.mu;
    sigma = normalizationParams{targetUser}.sigma;
    XTest = (XTest - mu) ./ sigma;

    % Predict
    net = models{targetUser};

    tic;
    yPredProb    = net(XTest')';
    inferenceTime = toc;
    throughput    = size(XTest, 1) / inferenceTime;

    % Store time + throughput
    userPerformance(targetUser, 1) = userPerformance(targetUser, 1) + inferenceTime;
    userPerformance(targetUser, 3) = throughput;

    % Similarities (user label vs probability)
    modelUserSimilarities = [testUserLabels, yPredProb];

    % Threshold
    threshold = 0.5;
    yPredBin  = double(yPredProb > threshold);

    % Metrics
    [metrics, confusionMat, Xroc, Yroc, Troc, AUC, EER, FAR, FRR] = ...
        calculatePerformanceMetrics_FAST(yTest, yPredBin, yPredProb, PLOT_ROC_AND_CONFUSION);

    trainSetSize         = size(userData(targetUser).trainFeatures,1);
    trainTargetSamples   = trainTargetSampleCount;
    trainImposterSamples = trainSetSize*(1/TrainTargetImposterRatio);
    testSetSize          = size(XTest,1);

    % Similarity stats for heatmap
    similarity_means          = zeros(1, numUsers);
    similarity_mids           = zeros(1, numUsers);
    similarity_mid_variations = zeros(1, numUsers);

    for u = userRange_min:userRange_max
        idx_u = (modelUserSimilarities(:,1) == u);
        if any(idx_u)
            vals = modelUserSimilarities(idx_u, 2);
            similarity_means(1,u) = mean(vals);
            mn = min(vals); mx = max(vals);
            similarity_mids(1,u) = (mx + mn)/2;
            similarity_mid_variations(1,u) = mx - similarity_mids(1,u);
        end
    end

    userSimilarityData(1, targetUser, :) = num2cell(similarity_means);
    userSimilarityData(2, targetUser, :) = num2cell(similarity_mids);
    userSimilarityData(3, targetUser, :) = num2cell(similarity_mid_variations);

    % metrics = [accuracy precision recall specificity f1 mcc FAR FRR EER AUC]
    userMetrics(targetUser, :) = [ ...
        metrics(1), metrics(2), metrics(3), metrics(4), ...
        metrics(5), metrics(6), metrics(7), metrics(8), ...
        metrics(9), AUC, ...
        trainSetSize, trainTargetSamples, trainImposterSamples, ...
        testSetSize];

    fprintf('  Accuracy: %.2f%%\n', userMetrics(targetUser,1)*100);
    fprintf('  Precision: %.2f%%\n', userMetrics(targetUser,2)*100);
    fprintf('  Recall: %.2f%%\n', userMetrics(targetUser,3)*100);
    fprintf('  Specificity: %.2f%%\n', userMetrics(targetUser,4)*100);
    fprintf('  F1-Score: %.2f%%\n', userMetrics(targetUser,5)*100);
    fprintf('  MCC: %.4f\n', userMetrics(targetUser,6));
    fprintf('  FAR: %.2f%%\n', userMetrics(targetUser,7));
    fprintf('  FRR: %.2f%%\n', userMetrics(targetUser,8));
    fprintf('  EER: %.2f%%\n', userMetrics(targetUser,9));
    fprintf('  AUC: %.4f\n', userMetrics(targetUser,10));
    fprintf('  Total Time (feat+train+test): %.2f s\n', userPerformance(targetUser,1));
    fprintf('  Memory Usage: %.2f MB\n', userPerformance(targetUser,2));
    fprintf('  Throughput: %.2f samples/s\n', userPerformance(targetUser,3));
end

%% ===================== SUMMARY & SAVING =================================

avgMetrics     = mean(userMetrics, 1, 'omitnan');
avgPerformance = mean(userPerformance, 1, 'omitnan');

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
    'AvgThroughput',         avgPerformance(3));

fprintf('\n==== Neural Network Architecture (FAST no-LOU) ====\n');
fprintf('Input Layer: variable (GA-selected features)\n');
for i = 1:length(hiddenLayerSizes)
    fprintf('Hidden Layer %d: %d neurons (%s)\n', ...
        i, hiddenLayerSizes(i), hiddenLayerActivationFcns{i});
end
fprintf('Output Layer: 1 neuron (%s)\n', outputLayerActivationFcn);
fprintf('Training Algorithm: %s\n', trainFcn);
fprintf('Performance Function: %s\n', performanceFcn);
fprintf('Max Epochs: %d\n', maxEpochs);

fprintf('\n==== Performance Benchmarks ====\n');
fprintf('Average Total Time: %.2f s (±%.2f)\n', ...
    mean(userPerformance(:,1)), std(userPerformance(:,1)));
fprintf('Average Memory Usage: %.2f MB (±%.2f)\n', ...
    mean(userPerformance(:,2)), std(userPerformance(:,2)));
fprintf('Average Throughput: %.2f samples/s (±%.2f)\n', ...
    mean(userPerformance(:,3)), std(userPerformance(:,3)));

summaryTable = table((1:numUsers)', ...
    userPerformance(:,1), ...
    userPerformance(:,2), ...
    userPerformance(:,3), ...
    userMetrics(:,1)*100, ...
    userMetrics(:,2)*100, ...
    userMetrics(:,3)*100, ...
    userMetrics(:,4)*100, ...
    userMetrics(:,5)*100, ...
    userMetrics(:,6), ...
    userMetrics(:,7), ...
    userMetrics(:,8), ...
    userMetrics(:,9), ...
    userMetrics(:,10), ...
    'VariableNames', { ...
    'User', 'TotalTime_sec', 'MemoryUsage_MB', 'Throughput_samples_per_sec', ...
    'Accuracy', 'Precision', 'Recall', 'Specificity', 'F1_Score', ...
    'MCC', 'FAR', 'FRR', 'EER', 'AUC'});

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
    'Avg_MCC', 'Avg_FAR', 'Avg_FRR', 'Avg_EER', 'Avg_AUC'});

fprintf('\n==== Summary Table ====\n');
disp(summaryTable);
disp('Overall Metrics:');
disp(overallMetrics);

% Similarity heatmap
similarityMatrix = zeros(numUsers, numUsers);
labelStrings     = cell(numUsers, numUsers);

for i = 1:numUsers
    for j = 1:numUsers
        val = cell2mat(userSimilarityData(1,i,j));
        mid = cell2mat(userSimilarityData(2,i,j));
        var = cell2mat(userSimilarityData(3,i,j));
        if isempty(val), val = 0; end
        if isempty(mid), mid = 0; end
        if isempty(var), var = 0; end
        similarityMatrix(i,j) = val;
        labelStrings{i,j} = sprintf('%.2f\nM: %.2f\n(±%.3f)', val, mid, var);
    end
end

figure('Position', [100 100 800 600]);
imagesc(similarityMatrix);
colormap(sky); % if 'sky' not available, change to 'parula'
c = colorbar;
c.Label.String = 'Similarity Score';

[Xh, Yh] = meshgrid(1:numUsers, 1:numUsers);
for i = 1:numUsers
    for j = 1:numUsers
        text(j, i, labelStrings{i,j}, ...
            'HorizontalAlignment', 'center', ...
            'Color', 'black', ...
            'FontSize', 9);
    end
end

set(gca, 'XTick', 1:numUsers, 'XTickLabel', userRange_min:userRange_max);
set(gca, 'YTick', 1:numUsers, 'YTickLabel', userRange_min:userRange_max);
xlabel("User N's similarity score");
ylabel("User N's Model");
title('User similarity scores for each user model (no LOU)');
axis square;

save('benchmark_results_noLOU_fast.mat', 'summaryTable', 'overallMetrics', 'results');
save('user_authentication_models_noLOU_fast.mat', 'models', 'selectedFeaturesPerUser', 'normalizationParams');

%% ===================== LOCAL FUNCTIONS ==================================

function [selectedFeatures, featureSelectionTime] = ...
    performGeneticFeatureSelection_FAST_noLOU(userData, targetUser, ...
                                              userRange_min, userRange_max, ...
                                              MAX_FEATURES_FOR_GA, ...
                                              GA_PopSize, GA_MaxGen)
% FAST GA-based feature selection using ALL other users as imposters.

tic;

% Combine data for feature selection (target=1, others=0)
X = userData(targetUser).trainFeatures;
y = ones(size(X, 1), 1);

for imposterUser = userRange_min:userRange_max
    if imposterUser ~= targetUser
        Xi = userData(imposterUser).trainFeatures;
        X  = [X; Xi];
        y  = [y; zeros(size(Xi,1),1)];
    end
end

% Keep only first K features for GA (for speed) while remembering mapping
nAllFeatures = size(X,2);
if nAllFeatures > MAX_FEATURES_FOR_GA
    featureIdxForGA = 1:MAX_FEATURES_FOR_GA;
    Xga = X(:, featureIdxForGA);
else
    featureIdxForGA = 1:nAllFeatures;
    Xga = X;
end
nFeatures = size(Xga,2);

options = optimoptions('ga', ...
    'Display', 'off', ...
    'PopulationSize', GA_PopSize, ...
    'MaxGenerations', GA_MaxGen, ...
    'UseVectorized', false);

[selectedMask, ~] = ga(@(mask) featureSelectionEvalGA_FAST(mask, Xga, y), ...
    nFeatures, [], [], [], [], ...
    zeros(1,nFeatures), ones(1,nFeatures), [], options);

selectedLocal = find(selectedMask > 0.5);
if isempty(selectedLocal)
    selectedLocal = 1; % fallback
end

selectedFeatures   = featureIdxForGA(selectedLocal); % map back to original indices
featureSelectionTime = toc;
end

function score = featureSelectionEvalGA_FAST(mask, X, y)
% GA objective: cross-validated misclassification loss of linear SVM
mask = mask > 0.5;
if ~any(mask)
    score = 1; % worst
    return;
end

Xsel = X(:, mask);

try
    model = fitcsvm(Xsel, y, ...
        'KernelFunction', 'linear', ...
        'Standardize', true, ...
        'BoxConstraint', 1);
    cv    = crossval(model, 'KFold', 3); % smaller K for speed
    mcr   = kfoldLoss(cv);
    score = mcr;
catch
    score = 1;
end
end

function [metrics, confusionMat, X, Y, T, AUC, EER, FAR, FRR] = ...
    calculatePerformanceMetrics_FAST(yTrue, yPredBin, yPredProb, doPlots)

% Confusion matrix
confusionMat = confusionmat(yTrue, yPredBin);

TP = sum((yTrue==1) & (yPredBin==1));
TN = sum((yTrue==0) & (yPredBin==0));
FP = sum((yTrue==0) & (yPredBin==1));
FN = sum((yTrue==1) & (yPredBin==0));

accuracy   = (TP + TN) / max(TP+TN+FP+FN, eps);
precision  = TP / max(TP+FP, eps);
recall     = TP / max(TP+FN, eps);
specificity= TN / max(TN+FP, eps);
f1_score   = 2 * (precision*recall) / max(precision+recall, eps);

mcc_num = (TP*TN) - (FP*FN);
mcc_den = sqrt(max((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN), eps));
mcc     = mcc_num / mcc_den;

% ROC and AUC
[X, Y, T, AUC] = perfcurve(yTrue, yPredProb, 1);

FAR = X;          % FPR
FRR = 1 - Y;      % 1-TPR

[~, eerIdx] = min(abs(FAR - FRR));
EER = (FAR(eerIdx) + FRR(eerIdx)) / 2;

if doPlots
    figure;
    plot(FAR, Y, 'LineWidth', 2);
    hold on;
    plot(FAR(eerIdx), Y(eerIdx), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
    text(FAR(eerIdx), Y(eerIdx), sprintf('  EER = %.2f%%', EER*100), ...
        'VerticalAlignment','bottom');
    xlabel('False Positive Rate (FAR)');
    ylabel('True Positive Rate (1 - FRR)');
    title(sprintf('ROC Curve (AUC = %.3f)', AUC));
    grid on;
end

metrics = [accuracy, precision, recall, specificity, f1_score, ...
           mcc, FAR(eerIdx)*100, FRR(eerIdx)*100, EER*100, AUC];
end