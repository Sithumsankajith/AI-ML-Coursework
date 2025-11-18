%% FeedForwardNet_with_Overfitting_Prevention_LOU_EER.m
% Complete User Authentication using MLP Neural Network - Binary Classification
% LOU version with ANOVA+MI+SG feature selection, z-score normalization, EER threshold.

clear all;
close all;
clc;

rng(100);  % For reproducibility

% Define script params
userRange_min = 1;
userRange_max = 10;
numUsers      = userRange_max - userRange_min + 1;

% Overfitting prevention parameters
TrainTargetImposterRatio = 1/5;  % Fixed ratio 1:5
dropoutRate              = 0.3;  %#ok<NASGU>
l2RegParam               = 1e-4;
performanceGoal          = 1e-5;
minGrad                  = 1e-6;
earlyStoppingPatience    = 10;
maxEpochs                = 500;
learningRate             = 0.01;

%% 1. Data Loading and Preprocessing
filePatternsTrain = 'Acc_TimeD_FreqD_FDay';
filePatternsTest  = 'Acc_TimeD_FreqD_MDay';

fprintf('Loading data for each user...\n');

userData = struct('trainFeatures', [], 'testFeatures', []);
userData = repmat(userData, 1, userRange_max);

for user = userRange_min:userRange_max
    userStr  = sprintf('U%02d', user);

    trainBase = [userStr '_' filePatternsTrain '.mat'];
    testBase  = [userStr '_' filePatternsTest  '.mat'];

    trainFile = trainBase;
    testFile  = testBase;

    if ~exist(trainFile, 'file') || ~exist(testFile, 'file')
        trainFile = fullfile('dataset', trainBase);
        testFile  = fullfile('dataset', testBase);
    end

    if exist(trainFile, 'file') && exist(testFile, 'file')
        trainData = load(trainFile);
        testData  = load(testFile);

        userData(user).trainFeatures = trainData.(char(fieldnames(trainData)));
        userData(user).testFeatures  = testData.(char(fieldnames(testData)));

        [r, c] = size(userData(user).trainFeatures);
        fprintf('User %d: %d train samples x %d features\n', user, r, c);
    else
        fprintf('Missing data files for user %d\n', user);
        userData(user).trainFeatures = [];
        userData(user).testFeatures  = [];
    end
end

% Leave-Out Users (fixed, as before)
leaveOutUsersList = [6, 3, 2, 5, 6, 1, 9, 7, 7, 3];

% Feature Selection Parameters
anovaThreshold   = 0.05;
topFeaturePercent = 0.75;

selectedFeatures = cell(numUsers, 1);
featureAnalysis  = struct();

%% 2. Feature selection (ANOVA+MI+Grad) per user
for targetUser = userRange_min:userRange_max
    X = userData(targetUser).trainFeatures;
    if isempty(X)
        warning('No train data for user %d, skipping feature selection', targetUser);
        continue;
    end
    y = ones(size(X, 1), 1);
    for imposterUser = 1:numUsers
        if imposterUser ~= targetUser && imposterUser ~= leaveOutUsersList(targetUser)
            Xi = userData(imposterUser).trainFeatures;
            if isempty(Xi), continue; end
            X = [X; Xi];
            y = [y; zeros(size(Xi,1),1)];
        end
    end

    numFeatures = size(X, 2);

    % ANOVA
    pValues = zeros(1, numFeatures);
    for i = 1:numFeatures
        pValues(i) = anova1(X(:, i), y, 'off');
    end
    anovaSelected = find(pValues < anovaThreshold);

    % Mutual Information
    miScores = zeros(1, numFeatures);
    for i = 1:numFeatures
        miScores(i) = calculate_mutual_information(X(:, i), y);
    end
    [~, miRanking] = sort(miScores, 'descend');
    miSelected = miRanking(1:round(topFeaturePercent*numFeatures));

    % Steepest Gradient
    netFS = feedforwardnet(10, 'trainscg');
    netFS = train(netFS, X', y');
    gradients = abs(netFS.IW{1});
    [~, sgRanking] = sort(mean(gradients, 1), 'descend');
    sgSelected = sgRanking(1:round(topFeaturePercent*numFeatures));

    % Visualization + combined importance
    figure('Name', sprintf('Feature Analysis - User %d', targetUser));
    subplot(2,2,1);
    featureScores = zeros(numFeatures, 3);
    featureScores(:,1) = 1 - normalize(reshape(pValues, [], 1), 'range');
    featureScores(:,2) = normalize(reshape(miScores, [], 1), 'range');
    meanGradients = mean(gradients, 1)';
    if length(meanGradients) ~= numFeatures
        meanGradients = interp1(1:length(meanGradients), meanGradients, ...
                                linspace(1, length(meanGradients), numFeatures));
    end
    featureScores(:,3) = normalize(meanGradients, 'range');
    bar(featureScores, 'stacked');
    title(sprintf('Feature Importance by Method (%d features)', numFeatures));
    legend('ANOVA', 'MI', 'Gradient');
    xlabel('Feature Index'); ylabel('Normalized Importance');

    weights = [0.4, 0.3, 0.3];
    combinedScores = featureScores * weights';
    [~, sortedIdx] = sort(combinedScores, 'descend');
    selectedFeatureIdx = sortedIdx(1:round(topFeaturePercent*numFeatures));

    subplot(2,2,2);
    correlationMatrix = corr(X(:, selectedFeatureIdx));
    imagesc(correlationMatrix);
    colormap(jet); colorbar;
    title(sprintf('Feature Correlation Matrix\n(%d features)', length(selectedFeatureIdx)));

    subplot(2,2,3);
    topN = min(5, length(selectedFeatureIdx));
    topFeatures = selectedFeatureIdx(1:topN);
    boxData = [];
    groupLabels = {};
    for i = 1:topN
        featureValues = X(:, topFeatures(i));
        boxData = [boxData; featureValues];
        groupLabels = [groupLabels; repmat({sprintf('Feature %d', i)}, length(featureValues), 1)];
    end
    boxplot(boxData, groupLabels);
    title(sprintf('Top %d Features Distribution', topN));
    xlabel('Feature Index'); ylabel('Feature Value');
    grid on;

    subplot(2,2,4);
    methodContribution = sum(featureScores, 1);
    if all(isfinite(methodContribution))
        pie(methodContribution);
        title('Feature Selection Method Contribution');
        legend({'ANOVA', 'MI', 'Gradient'}, 'Location', 'eastoutside');
    end

    % Trend plot
    figure('Name', sprintf('Feature Importance Trends - User %d', targetUser));
    subplot(2,1,1);
    comb2 = mean(featureScores, 2);
    [sortedScores2, sortedIdx2] = sort(comb2, 'descend');
    bar(sortedScores2(1:min(20,end)));
    title('Top 20 Features by Combined Importance');
    xlabel('Feature Rank'); ylabel('Importance Score'); grid on;

    subplot(2,1,2);
    topK = min(20, length(sortedIdx2));
    topFeatureScores = featureScores(sortedIdx2(1:topK), :);
    bar(topFeatureScores, 'grouped');
    title('Method Contributions for Top Features');
    xlabel('Feature Rank'); ylabel('Score');
    legend('ANOVA', 'MI', 'Gradient'); grid on;

    featureAnalysis(targetUser).scores          = featureScores;
    featureAnalysis(targetUser).combinedScores  = combinedScores;
    featureAnalysis(targetUser).selectedFeatures = selectedFeatureIdx;
    featureAnalysis(targetUser).correlationMatrix = correlationMatrix;

    fprintf('\nFeature Selection Summary for User %d:\n', targetUser);
    fprintf('Top 5 features: %s\n', mat2str(selectedFeatureIdx(1:min(5,end))));
    fprintf('Average correlation: %.4f\n', ...
        mean(abs(correlationMatrix(triu(true(size(correlationMatrix)),1)))));
    fprintf('ANOVA selected: %d\n', length(anovaSelected));
    fprintf('MI selected: %d\n', length(miSelected));
    fprintf('SG selected: %d\n', length(sgSelected));
    fprintf('Combined unique features: %d\n', length(selectedFeatureIdx));

    selectedFeatures{targetUser} = selectedFeatureIdx;
end

%% 3. Training + Testing (one model per user)

models            = cell(numUsers, 1);
userMetrics       = zeros(numUsers, 15);
userPerformance   = zeros(numUsers, 3); % [time, memory, throughput]
userSimilarityData= cell(3, numUsers, numUsers);
PrecisionPerUser  = zeros(numUsers,1);  % to average later

for targetUser = userRange_min:userRange_max
    if isempty(userData(targetUser).trainFeatures)
        warning('Skipping user %d (no training data).', targetUser);
        continue;
    end

    selIdx = selectedFeatures{targetUser};

    %% Build training set (target vs imposters)
    trainTargetSampleCount = size(userData(targetUser).trainFeatures, 1);
    trainImposterSampleCount = trainTargetSampleCount*(1/TrainTargetImposterRatio);
    trainSamplesPerImposter  = floor(trainImposterSampleCount/(numUsers-1));

    XTrain_pos = userData(targetUser).trainFeatures(:, selIdx);
    yTrain_pos = ones(trainTargetSampleCount, 1);

    XTrain_neg = [];
    yTrain_neg = [];

    for imposterUser = 1:numUsers
        if imposterUser ~= targetUser && imposterUser ~= leaveOutUsersList(targetUser)
            if isempty(userData(imposterUser).trainFeatures), continue; end
            nAvail = size(userData(imposterUser).trainFeatures, 1);
            k = min(trainSamplesPerImposter, nAvail);
            idx = randperm(nAvail, k);
            XTrain_neg = [XTrain_neg;
                userData(imposterUser).trainFeatures(idx, selIdx)];
            yTrain_neg = [yTrain_neg; zeros(k,1)];
        end
    end

    XTrain_raw = [XTrain_pos; XTrain_neg];
    yTrain     = [yTrain_pos; yTrain_neg];

    % update actual imposter count
    actualImposterSamples = sum(yTrain == 0);
    trainImposterSampleCount = actualImposterSamples;

    % z-score train
    [XTrain, mu, sigma] = zscore(XTrain_raw);
    sigma(sigma == 0) = 1;

    %% Build test set
    testTargetSampleCount = size(userData(targetUser).testFeatures, 1);
    testImposterSampleCount = 324;  % as in your original
    testSamplesPerImposter  = floor(testImposterSampleCount/(numUsers-1));

    XTest_pos = userData(targetUser).testFeatures(:, selIdx);
    yTest_pos = ones(testTargetSampleCount,1);

    XTest_neg = [];
    yTest_neg = [];
    testUserLabels = ones(testTargetSampleCount,1)*targetUser;

    for imposterUser = 1:numUsers
        if imposterUser ~= targetUser
            if isempty(userData(imposterUser).testFeatures), continue; end
            nAvail = size(userData(imposterUser).testFeatures, 1);
            k = min(testSamplesPerImposter, nAvail);
            idx = randperm(nAvail, k);
            XTest_neg = [XTest_neg;
                userData(imposterUser).testFeatures(idx, selIdx)];
            yTest_neg = [yTest_neg; zeros(k,1)];
            testUserLabels = [testUserLabels; ones(k,1)*imposterUser];
        end
    end

    XTest_raw = [XTest_pos; XTest_neg];
    yTest     = [yTest_pos; yTest_neg];

    % z-score test with train params
    XTest = (XTest_raw - mu) ./ sigma;

    assert(sum(yTest==1) == testTargetSampleCount);
    assert(sum(yTest==0) == testImposterSampleCount);

    %% Create and configure the network
    net = feedforwardnet(96, 'trainscg'); % slightly smaller than 131
    net.userdata.note = "Feedforward NN with LOU + ANOVA/MI/Gradient features";
    net.userdata.trainTargetImposterRatio = sprintf("1:%d", round(1/TrainTargetImposterRatio));
    net.userdata.dropoutRate              = dropoutRate;
    net.userdata.l2RegParam               = l2RegParam;
    net.userdata.performanceGoal          = performanceGoal;
    net.userdata.minGrad                  = minGrad;
    net.userdata.earlyStoppingPatience    = earlyStoppingPatience;
    net.userdata.maxEpochs                = maxEpochs;
    net.userdata.learningRate             = learningRate;
    net.userdata.targetUser               = sprintf('User %d', targetUser);
    net.performFcn                        = 'crossentropy';

    net.layers{1}.transferFcn  = 'logsig';
    net.layers{end}.transferFcn = 'tansig';

    net.trainParam.epochs   = maxEpochs;
    net.trainParam.goal     = performanceGoal;
    net.trainParam.min_grad = minGrad;
    net.performParam.regularization = l2RegParam;
    net.trainParam.max_fail = earlyStoppingPatience;
    net.trainParam.lr       = learningRate;
    net.trainParam.showWindow      = false;
    net.trainParam.showCommandLine = false;

    % Train
    tic;
    [net, ~] = train(net, XTrain', yTrain');
    trainTime = toc;

    models{targetUser} = net;
    modelInfo = whos('net');
    memoryUsage = modelInfo.bytes / (1024^2);

    %% Evaluate
    tic;
    yScores = net(XTest')';
    inferenceTime = toc;
    throughput = size(XTest,1)/inferenceTime;

    yTrueBin = (yTest == 1);

    [FPR, TPR, Thr, AUC] = perfcurve(yTrueBin, yScores, 1);
    FAR = FPR;
    FRR = 1 - TPR;
    [~, eerIdx] = min(abs(FAR - FRR));
    EER = (FAR(eerIdx) + FRR(eerIdx))/2;
    bestThr = Thr(eerIdx);

    yPredBin = double(yScores >= bestThr);

    % Confusion components
    tp = sum(yPredBin==1 & yTrueBin==1);
    tn = sum(yPredBin==0 & yTrueBin==0);
    fp = sum(yPredBin==1 & yTrueBin==0);
    fn = sum(yPredBin==0 & yTrueBin==1);

    genuinePrecision  = tp/(tp+fp+eps);
    impostorPrecision = tn/(tn+fn+eps);
    overallPrecision  = (genuinePrecision*sum(yTrueBin==1) + ...
                         impostorPrecision*sum(yTrueBin==0))/length(yTrueBin);

    PrecisionPerUser(targetUser) = overallPrecision;

    recall      = tp/(tp+fn+eps);
    specificity = tn/(tn+fp+eps);
    accuracy    = (tp+tn)/(tp+tn+fp+fn+eps);
    f1_score    = 2*(overallPrecision*recall)/(overallPrecision+recall+eps);

    fpr = fp/(fp+tn+eps);
    fnr = fn/(fn+tp+eps);
    mcc = ((tp*tn)-(fp*fn))/sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)+eps);

    % Similarity stats using binary predictions as similarity (you could also keep scores)
    modelUserSimilarities = [testUserLabels, yScores];

    similarity_means          = zeros(1, numUsers);
    similarity_mids           = zeros(1, numUsers);
    similarity_mid_variations = zeros(1, numUsers);

    for u = userRange_min:userRange_max
        idx = (modelUserSimilarities(:,1) == u);
        vals = modelUserSimilarities(idx, 2);
        similarity_means(1,u) = mean(vals);
        if ~isempty(vals)
            mn = min(vals); mx = max(vals);
            similarity_mids(1,u) = (mx + mn)/2;
            similarity_mid_variations(1,u) = mx - similarity_mids(1,u);
        end
    end

    userSimilarityData(1, targetUser, :) = num2cell(similarity_means);
    userSimilarityData(2, targetUser, :) = num2cell(similarity_mids);
    userSimilarityData(3, targetUser, :) = num2cell(similarity_mid_variations);

    userPerformance(targetUser,:) = [trainTime+inferenceTime, memoryUsage, throughput];

    userMetrics(targetUser,:) = [ ...
        accuracy, overallPrecision, recall, specificity, f1_score, ...
        mcc, fpr*100, fnr*100, EER*100, AUC, overallPrecision, ...
        size(XTrain,1), trainTargetSampleCount, trainImposterSampleCount, ...
        size(XTest,1)];

    fprintf('\nUser %d Results:\n', targetUser);
    fprintf('Accuracy: %.2f%%\n', accuracy*100);
    fprintf('Overall Precision: %.2f%%\n', overallPrecision*100);
    fprintf('Recall: %.2f%%\n', recall*100);
    fprintf('Specificity: %.2f%%\n', specificity*100);
    fprintf('F1-Score: %.2f%%\n', f1_score*100);
    fprintf('MCC: %.4f\n', mcc);
    fprintf('FAR: %.2f%%\n', fpr*100);
    fprintf('FRR: %.2f%%\n', fnr*100);
    fprintf('EER: %.2f%%\n', EER*100);
    fprintf('AUC: %.4f\n', AUC);
    fprintf('Training Time: %.4f s\n', trainTime);
    fprintf('Memory Usage: %.2f MB\n', memoryUsage);
    fprintf('Throughput: %.2f samples/s\n', throughput);

    % Confusion chart
    figure('Name', sprintf('Detailed Confusion Matrix - User %d', targetUser));
    cm = confusionchart(yTrueBin, yPredBin);
    cm.Title = sprintf('Confusion Matrix - User %d\nTP=%d, TN=%d, FP=%d, FN=%d', ...
        targetUser, tp, tn, fp, fn);
    cm.RowSummary = 'row-normalized';
    cm.ColumnSummary = 'column-normalized';

    % ROC
    figure;
    plot(FPR, TPR, 'LineWidth', 2);
    hold on;
    plot(FAR(eerIdx), TPR(eerIdx), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
    xlabel('False Positive Rate'); ylabel('True Positive Rate');
    title(sprintf('ROC Curve - User %d (AUC = %.3f)', targetUser, AUC));
    grid on;
end

%% Averages
avgMetrics      = mean(userMetrics, 1);
avgPerformance  = mean(userPerformance, 1);
avgOverallPrecision = mean(PrecisionPerUser);

results = struct( ...
  'Ratio', '1:5', ...
  'AvgAccuracy',           avgMetrics(1)*100, ...
  'AvgOverallPrecision',   avgOverallPrecision*100, ...
  'AvgRecall',             avgMetrics(3)*100, ...
  'AvgSpecificity',        avgMetrics(4)*100, ...
  'AvgF1Score',            avgMetrics(5)*100, ...
  'AvgMCC',                avgMetrics(6), ...
  'AvgFAR',                avgMetrics(7), ...
  'AvgFRR',                avgMetrics(8), ...
  'AvgEER',                avgMetrics(9), ...
  'AvgAUC',                avgMetrics(10), ...
  'AvgTrainingSetSize',    avgMetrics(12), ...
  'AvgTrainTargetSamples', avgMetrics(13), ...
  'AvgTrainImposterSamples', avgMetrics(14), ...
  'AvgTestSetSize',        avgMetrics(15), ...
  'AvgTotalTime',          avgPerformance(1), ...
  'AvgMemoryUsage',        avgPerformance(2), ...
  'AvgThroughput',         avgPerformance(3));

fprintf('\n==== Neural Network Architecture ====\n');
fprintf('Input Layer: %d neurons\n', size(XTrain, 2));
fprintf('Hidden Layer 1: 96 neurons (logsig)\n');
fprintf('Output Layer: 1 neuron (tansig)\n');
fprintf('Training Algorithm: trainscg\n');
fprintf('Performance Function: Cross-Entropy\n');
fprintf('L2 Regularization: %e\n', l2RegParam);
fprintf('Max Epochs: %d\n', maxEpochs);

fprintf('\n==== Performance Benchmarks ====\n');
fprintf('Average Training+Inference Time: %.4f s (±%.4f)\n', ...
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
  'VariableNames', {...
  'User', 'TotalTime_sec', 'MemoryUsage_MB', 'Throughput_samples_per_sec', ...
  'Accuracy', 'OverallPrecision', 'Recall', 'Specificity', 'F1_Score', ...
  'MCC', 'FAR', 'FRR', 'EER', 'AUC'});

overallMetrics = table(mean(userPerformance(:,1)), ...
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
  'VariableNames', {...
  'Avg_TotalTime_sec', 'Avg_MemoryUsage_MB', 'Avg_Throughput_samples_per_sec', ...
  'Avg_Accuracy', 'Avg_OverallPrecision', 'Avg_Recall', 'Avg_Specificity', 'Avg_F1_Score', ...
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
    similarityMatrix(i,j) = val;
    labelStrings{i,j} = sprintf('%.2f\nM: %.2f\n(±%.3f)', val, mid, var);
  end
end

figure('Position', [100 100 800 600]);
imagesc(similarityMatrix);
colormap(sky); % or parula
c = colorbar;
c.Label.String = 'Similarity Score';

[Xg,Yg] = meshgrid(1:numUsers, 1:numUsers);
for i = 1:numUsers
  for j = 1:numUsers
    text(i, j, labelStrings{j,i}, ...
      'HorizontalAlignment', 'center', ...
      'Color', 'black', ...
      'FontSize', 10);
  end
end

set(gca, 'XTick', 1:numUsers, 'XTickLabel', userRange_min:userRange_max);
set(gca, 'YTick', 1:numUsers, 'YTickLabel', userRange_min:userRange_max);
xlabel("User N's similarity score");
ylabel("User N's Model");
title('User similarity scores for each user model');
axis square;

save('benchmark_results_LOU_EER.mat', 'summaryTable', 'overallMetrics');
save('user_authentication_models_LOU_EER.mat', 'models');
save('feature_analysis_results_LOU_EER.mat', 'featureAnalysis');

%% Helper: mutual information
function mi = calculate_mutual_information(x, y)
    x = (x - min(x)) / (max(x) - min(x) + eps);
    nbins = 10;
    edges = linspace(0, 1, nbins+1);
    [~, disc_x] = histc(x, edges);
    disc_x(disc_x == nbins+1) = nbins;

    joint_hist = zeros(nbins, 2);
    for i = 1:length(x)
        if disc_x(i) > 0
            joint_hist(disc_x(i), y(i)+1) = joint_hist(disc_x(i), y(i)+1) + 1;
        end
    end

    joint_p = joint_hist / (length(x) + eps);
    p_x = sum(joint_p, 2);
    p_y = sum(joint_p, 1);

    mi = 0;
    for i = 1:nbins
        for j = 1:2
            if joint_p(i,j) > 0
                mi = mi + joint_p(i,j) * ...
                    log2(joint_p(i,j) / (p_x(i) * p_y(j) + eps) + eps);
            end
        end
    end
end
