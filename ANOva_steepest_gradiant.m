%% FeedForwardNet with Overfitting Prevention + Feature Selection %%%%%
% Complete User Authentication using MLP Neural Network - Binary Classification

clear all; close all; clc;
rng(100);  % For reproducibility

% Define script params
userRange_min = 1;
userRange_max = 10;
numUsers      = userRange_max - userRange_min + 1;

% Overfitting prevention parameters
TrainTargetImposterRatio = 1/5;   % Fixed ratio 1:5
dropoutRate              = 0.3;   % (defined but not used by feedforwardnet)
l2RegParam               = 1e-4;  % L2 regularization parameter
performanceGoal          = 1e-5;  % Performance goal for training
minGrad                  = 1e-6;  % Minimum gradient for training
earlyStoppingPatience    = 10;    % Patience for early stopping
maxEpochs                = 500;   % Maximum number of training epochs
learningRate             = 0.01;  % Learning rate

%% 1. Data Loading and Preprocessing
filePatternsTrain = 'Acc_TimeD_FreqD_FDay';
filePatternsTest  = 'Acc_TimeD_FreqD_MDay';

fprintf('Loading data for each user...\n');

% Preallocate storage datasets
userData = struct('trainFeatures', [], 'testFeatures', []);
userData = repmat(userData, 1, userRange_max);

for user = userRange_min:userRange_max
    userStr  = sprintf('U%02d', user);
    trainFile = fullfile('dataset', [userStr '_' filePatternsTrain '.mat']);
    testFile  = fullfile('dataset', [userStr '_' filePatternsTest  '.mat']);

    if exist(trainFile, 'file') && exist(testFile, 'file')
        trainData = load(trainFile);
        testData  = load(testFile);

        userData(user).trainFeatures = trainData.(char(fieldnames(trainData)));
        userData(user).testFeatures  = testData.(char(fieldnames(testData)));

        [r, c] = size(userData(user).trainFeatures);
        fprintf('User %d: %d train samples x %d features\n', user, r, c);
    else
        fprintf('Missing data files for user %d\n', user);
        fprintf('  Tried train: %s\n', trainFile);
        fprintf('  Tried test : %s\n', testFile);
    end
end

% Pre-defined leave-out users list (for imposters)
leaveOutUsersList = [6, 3, 2, 5, 6, 1, 9, 7, 7, 3];

% Feature Selection Parameters
anovaThreshold   = 0.05;  % Threshold for ANOVA p-values
topFeaturePercent = 0.75; % Top 75% features to select

selectedFeatures = cell(numUsers, 1);
featureAnalysis  = struct();

%% 2. Feature Selection per User (ANOVA + MI + Gradient)
for targetUser = userRange_min:userRange_max
    if isempty(userData(targetUser).trainFeatures)
        fprintf('User %d has no training data for feature selection. Skipping.\n', targetUser);
        continue;
    end

    % Combine target vs imposters (excluding leave-out imposters)
    X = userData(targetUser).trainFeatures;
    y = ones(size(X, 1), 1);
    for imposterUser = 1:numUsers
        if imposterUser ~= targetUser && imposterUser ~= leaveOutUsersList(targetUser) ...
                && ~isempty(userData(imposterUser).trainFeatures)
            X = [X; userData(imposterUser).trainFeatures];
            y = [y; zeros(size(userData(imposterUser).trainFeatures, 1), 1)];
        end
    end

    numFeatures = size(X, 2);

    %% ANOVA Feature Selection
    pValues = zeros(1, numFeatures);
    for i = 1:numFeatures
        pValues(i) = anova1(X(:, i), y, 'off');
    end
    anovaSelected = find(pValues < anovaThreshold);

    %% Mutual Information Feature Selection
    miScores = zeros(1, numFeatures);
    for i = 1:numFeatures
        miScores(i) = calculate_mutual_information(X(:, i), y);
    end
    [~, miRanking] = sort(miScores, 'descend');
    miSelected = miRanking(1:round(topFeaturePercent*numFeatures));

    %% Steepest Gradient Feature Selection (small helper NN)
    netFS = feedforwardnet(10, 'trainscg');
    netFS.trainParam.showWindow = false;
    netFS.trainParam.showCommandLine = false;
    netFS = train(netFS, X', y');
    gradients = abs(netFS.IW{1});
    meanGradients = mean(gradients, 1)';

    if length(meanGradients) ~= numFeatures
        % Interpolate gradient scores if dimensions don't match
        meanGradients = interp1(1:length(meanGradients), meanGradients, ...
                                linspace(1, length(meanGradients), numFeatures));
    end
    [~, sgRanking] = sort(meanGradients, 'descend');
    sgSelected = sgRanking(1:round(topFeaturePercent*numFeatures));

    %% Build normalized feature importance matrix
    featureScores = zeros(numFeatures, 3);
    % ANOVA: lower p => more important
    featureScores(:,1) = 1 - normalize(pValues(:), 'range');
    featureScores(:,2) = normalize(miScores(:), 'range');
    featureScores(:,3) = normalize(meanGradients(:), 'range');

    % Combined weighted score
    weights        = [0.4, 0.3, 0.3];
    combinedScores = featureScores * weights';
    [sortedScores, sortedIdx] = sort(combinedScores, 'descend'); %#ok<ASGLU>

    selectedFeatureIdx = sortedIdx(1:round(topFeaturePercent*numFeatures));

    % Correlation matrix over selected features
    correlationMatrix = corr(X(:, selectedFeatureIdx));

    %% (Optional) Plots – keep for report
    figure('Name', sprintf('Feature Analysis - User %d', targetUser));

    % 1) Stacked importance
    subplot(2,2,1);
    bar(featureScores, 'stacked');
    title(sprintf('Feature Importance by Method (%d features)', numFeatures));
    legend('ANOVA', 'MI', 'Gradient');
    xlabel('Feature Index'); ylabel('Normalized Importance');

    % 2) Correlation matrix
    subplot(2,2,2);
    imagesc(correlationMatrix);
    colormap(jet); colorbar;
    title(sprintf('Feature Correlation Matrix (%d features)', length(selectedFeatureIdx)));

    % 3) Boxplots for top 5 features
    subplot(2,2,3);
    topN       = min(5, length(selectedFeatureIdx));
    topFeatures = selectedFeatureIdx(1:topN);
    boxData    = X(:, topFeatures);
    boxplot(boxData, 'Labels', arrayfun(@(i) sprintf('F%d', i), 1:topN, 'UniformOutput', false));
    title(sprintf('Top %d Features Distribution', topN));
    xlabel('Feature'); ylabel('Value'); grid on;

    % 4) Method contribution pie
    subplot(2,2,4);
    methodContribution = sum(featureScores, 1);
    if all(isfinite(methodContribution))
        pie(methodContribution);
        title('Feature Selection Method Contribution');
        legend({'ANOVA', 'MI', 'Gradient'}, 'Location', 'eastoutside');
    end

    % Additional feature importance trends
    figure('Name', sprintf('Feature Importance Trends - User %d', targetUser));
    subplot(2,1,1);
    avgScores = mean(featureScores, 2);
    [sortedAvg, sortedIdx2] = sort(avgScores, 'descend');
    bar(sortedAvg(1:min(20,end)));
    title('Top 20 Features by Combined Importance');
    xlabel('Feature Rank'); ylabel('Importance'); grid on;

    subplot(2,1,2);
    topK = min(20, length(sortedIdx2));
    topFeatureScores = featureScores(sortedIdx2(1:topK), :);
    bar(topFeatureScores, 'grouped');
    title('Method Contributions for Top Features');
    xlabel('Feature Rank'); ylabel('Score');
    legend('ANOVA', 'MI', 'Gradient'); grid on;

    %% Store feature analysis
    featureAnalysis(targetUser).pValues          = pValues;
    featureAnalysis(targetUser).miScores         = miScores;
    featureAnalysis(targetUser).gradientScores   = meanGradients';
    featureAnalysis(targetUser).scores           = featureScores;
    featureAnalysis(targetUser).combinedScores   = combinedScores;
    featureAnalysis(targetUser).selectedFeatures = selectedFeatureIdx;
    featureAnalysis(targetUser).correlationMatrix = correlationMatrix;

    selectedFeatures{targetUser} = selectedFeatureIdx;

    fprintf('\nFeature Selection Summary for User %d:\n', targetUser);
    fprintf('Top 5 selected features: %s\n', mat2str(selectedFeatureIdx(1:min(5,end))));
    fprintf('ANOVA selected: %d\n', length(anovaSelected));
    fprintf('MI selected:    %d\n', length(miSelected));
    fprintf('SG selected:    %d\n', length(sgSelected));
    fprintf('Combined unique features: %d\n', length(selectedFeatureIdx));
    fprintf('Avg |correlation| among selected: %.4f\n', ...
        mean(abs(correlationMatrix(triu(true(size(correlationMatrix)),1)))));
end

%% 3. Model Training and Evaluation (one-vs-all)

models            = cell(numUsers, 1);
userMetrics       = zeros(numUsers, 14); % [10 metrics + 4 size stats]
userPerformance   = zeros(numUsers, 3);  % [totalTime, memoryMB, throughput]
userSimilarityData = cell(3, numUsers, numUsers);

for targetUser = userRange_min:userRange_max
    if isempty(userData(targetUser).trainFeatures) || isempty(selectedFeatures{targetUser})
        fprintf('\nUser %d: Missing train data or selected features. Skipping model.\n', targetUser);
        continue;
    end

    fprintf('\n===== Training model for User %d =====\n', targetUser);

    selIdx = selectedFeatures{targetUser};

    % Training set
    trainTargetSampleCount   = size(userData(targetUser).trainFeatures, 1);
    trainImposterSampleCount = trainTargetSampleCount * (1/TrainTargetImposterRatio);
    trainSamplesPerImposter  = max(floor(trainImposterSampleCount/(numUsers-1)), 1);

    XTrain = userData(targetUser).trainFeatures(:, selIdx);
    yTrain = ones(trainTargetSampleCount, 1);

    trainImposterFeatures = [];
    trainImposterLabels   = [];
    for imposterUser = 1:numUsers
        if imposterUser ~= targetUser && imposterUser ~= leaveOutUsersList(targetUser) ...
                && ~isempty(userData(imposterUser).trainFeatures)
            available = size(userData(imposterUser).trainFeatures, 1);
            k         = min(trainSamplesPerImposter, available);
            idx       = randperm(available, k);
            trainImposterFeatures = [trainImposterFeatures;
                userData(imposterUser).trainFeatures(idx, selIdx)];
            trainImposterLabels = [trainImposterLabels; zeros(k, 1)];
        end
    end

    if isempty(trainImposterFeatures)
        fprintf('  User %d: no imposter training samples. Skipping.\n', targetUser);
        continue;
    end

    XTrain = [XTrain; trainImposterFeatures];
    yTrain = [yTrain; trainImposterLabels];

    % Verify ratio
    fprintf('  Train positives: %d, negatives: %d\n', sum(yTrain==1), sum(yTrain==0));

    % Testing set
    if isempty(userData(targetUser).testFeatures)
        fprintf('  User %d has no test data. Skipping.\n', targetUser);
        continue;
    end

    testTargetSampleCount   = size(userData(targetUser).testFeatures, 1);
    testImposterSampleCount = 324; % fixed as per original script
    testSamplesPerImposter  = max(floor(testImposterSampleCount/(numUsers-1)), 1);

    XTest = userData(targetUser).testFeatures(:, selIdx);
    yTest = ones(testTargetSampleCount, 1);
    testUserLabels = ones(testTargetSampleCount, 1) * targetUser;

    testImposterFeatures = [];
    testImposterLabels   = [];

    for imposterUser = 1:numUsers
        if imposterUser ~= targetUser && ~isempty(userData(imposterUser).testFeatures)
            available = size(userData(imposterUser).testFeatures, 1);
            k         = min(testSamplesPerImposter, available);
            idx       = randperm(available, k);
            testImposterFeatures = [testImposterFeatures;
                userData(imposterUser).testFeatures(idx, selIdx)];
            testImposterLabels = [testImposterLabels; zeros(k, 1)];
            testUserLabels     = [testUserLabels; imposterUser * ones(k,1)];
        end
    end

    if isempty(testImposterFeatures)
        fprintf('  User %d: no imposter test samples. Skipping.\n', targetUser);
        continue;
    end

    XTest = [XTest; testImposterFeatures];
    yTest = [yTest; testImposterLabels];

    fprintf('  Test positives: %d, negatives: %d\n', sum(yTest==1), sum(yTest==0));

    %% Neural Network definition
    net = feedforwardnet(57, 'trainscg');
    net.userdata.note                     = "Feedforward NN with ANOVA+MI+Gradient features (LOU)";
    net.userdata.trainTargetImposterRatio = sprintf("1:%d", round(1/TrainTargetImposterRatio));
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

    % Train network
    tic;
    [net, tr] = train(net, XTrain', yTrain');
    trainTime = toc;

    models{targetUser} = net;

    modelInfo   = whos('net');
    memoryUsage = modelInfo.bytes / (1024^2); % MB

    %% Evaluation
    tic;
    yPredProb     = net(XTest')';
    inferenceTime = toc;
    throughput    = size(XTest, 1) / max(inferenceTime, eps);

    userPerformance(targetUser,:) = [trainTime + inferenceTime, memoryUsage, throughput];

    % Binary decisions
    yPredBin = double(yPredProb > 0.5);

    % Confusion matrix terms
    tp = sum(yPredBin == 1 & yTest == 1);
    tn = sum(yPredBin == 0 & yTest == 0);
    fp = sum(yPredBin == 1 & yTest == 0);
    fn = sum(yPredBin == 0 & yTest == 1);

    % Metrics
    epsVal     = 1e-12;
    precision  = tp / max(tp + fp, epsVal);
    recall     = tp / max(tp + fn, epsVal);
    specificity= tn / max(tn + fp, epsVal);
    accuracy   = (tp + tn) / max(tp + tn + fp + fn, epsVal);
    f1_score   = 2 * precision * recall / max(precision + recall, epsVal);

    fpr = fp / max(fp + tn, epsVal);
    fnr = fn / max(fn + tp, epsVal);
    eer = (fpr + fnr) / 2;

    mcc_num = (tp * tn) - (fp * fn);
    mcc_den = sqrt(max((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn), epsVal));
    mcc     = mcc_num / max(mcc_den, epsVal);

    % ROC & AUC (use continuous outputs)
    [Xroc,Yroc,~,AUC] = perfcurve(yTest, yPredProb, 1);

    % Store similarities using probabilities
    modelUserSimilarities = [testUserLabels, yPredProb];

    % Similarity statistics
    similarity_means          = zeros(1, numUsers);
    similarity_mids           = zeros(1, numUsers);
    similarity_mid_variations = zeros(1, numUsers);

    for u = userRange_min:userRange_max
        idxu = (modelUserSimilarities(:,1) == u);
        if any(idxu)
            vals = modelUserSimilarities(idxu,2);
            similarity_means(1,u) = mean(vals);
            vmin = min(vals); vmax = max(vals);
            similarity_mids(1,u)           = (vmax + vmin)/2;
            similarity_mid_variations(1,u) = vmax - similarity_mids(1,u);
        end
    end

    userSimilarityData(1, targetUser, :) = num2cell(similarity_means);
    userSimilarityData(2, targetUser, :) = num2cell(similarity_mids);
    userSimilarityData(3, targetUser, :) = num2cell(similarity_mid_variations);

    % Store metrics row
    userMetrics(targetUser,:) = [ ...
        accuracy, precision, recall, specificity, ...
        f1_score, mcc, fpr*100, fnr*100, eer*100, AUC, ...
        size(XTrain,1), trainTargetSampleCount, trainImposterSampleCount, ...
        size(XTest,1) ...
    ];

    % Display per-user summary
    fprintf('\n==== Individual User Performance ====\n');
    fprintf('User %d Results:\n', targetUser);
    fprintf('Accuracy:   %.2f%%\n', accuracy*100);
    fprintf('Precision:  %.2f%%\n', precision*100);
    fprintf('Recall:     %.2f%%\n', recall*100);
    fprintf('Specificity:%.2f%%\n', specificity*100);
    fprintf('F1-Score:   %.2f%%\n', f1_score*100);
    fprintf('MCC:        %.4f\n', mcc);
    fprintf('FAR:        %.2f%%\n', fpr*100);
    fprintf('FRR:        %.2f%%\n', fnr*100);
    fprintf('EER:        %.2f%%\n', eer*100);
    fprintf('AUC:        %.4f\n', AUC);
    fprintf('Training+Inference Time: %.4f s\n', trainTime + inferenceTime);
    fprintf('Memory Usage:            %.2f MB\n', memoryUsage);
    fprintf('Throughput:              %.2f samples/s\n', throughput);

    % Confusion chart
    figure('Name', sprintf('Confusion Matrix - User %d', targetUser));
    cm = confusionchart(yTest, yPredBin);
    cm.Title = sprintf('Confusion Matrix - User %d', targetUser);
    cm.RowSummary = 'row-normalized';
    cm.ColumnSummary = 'column-normalized';

    % ROC curve
    figure('Name', sprintf('ROC - User %d', targetUser));
    plot(Xroc,Yroc);
    xlabel('False Positive Rate'); ylabel('True Positive Rate');
    title(sprintf('ROC Curve - User %d (AUC = %.3f)', targetUser, AUC));
    grid on;
end

%% 4. Averages & Summary

avgMetrics     = mean(userMetrics, 1, 'omitnan');
avgPerformance = mean(userPerformance, 1, 'omitnan');

results = struct( ...
  'Ratio',                  '1:5', ...
  'AvgAccuracy',            avgMetrics(1)*100, ...
  'AvgPrecision',           avgMetrics(2)*100, ...
  'AvgRecall',              avgMetrics(3)*100, ...
  'AvgSpecificity',         avgMetrics(4)*100, ...
  'AvgF1Score',             avgMetrics(5)*100, ...
  'AvgMCC',                 avgMetrics(6), ...
  'AvgFAR',                 avgMetrics(7), ...
  'AvgFRR',                 avgMetrics(8), ...
  'AvgEER',                 avgMetrics(9), ...
  'AvgAUC',                 avgMetrics(10), ...
  'AvgTrainingSetSize',     avgMetrics(11), ...
  'AvgTrainTargetSamples',  avgMetrics(12), ...
  'AvgTrainImposterSamples',avgMetrics(13), ...
  'AvgTestSetSize',         avgMetrics(14), ...
  'AvgTotalTime',           avgPerformance(1), ...
  'AvgMemoryUsage',         avgPerformance(2), ...
  'AvgThroughput',          avgPerformance(3));

fprintf('\n==== Neural Network Architecture ====\n');
fprintf('Input Layer: %d neurons (selected features)\n', size(XTrain, 2));
fprintf('Hidden Layer: 57 neurons (logsig)\n');
fprintf('Output Layer: 1 neuron (tansig)\n');
fprintf('Training Algorithm: Scaled Conjugate Gradient (trainscg)\n');
fprintf('Performance Function: Cross-Entropy\n');
fprintf('L2 Regularization: %e\n', l2RegParam);
fprintf('Max Epochs: %d\n', maxEpochs);

fprintf('\n==== Performance Benchmarks ====\n');
fprintf('Average Total Time: %.4f s (±%.4f)\n', mean(userPerformance(:,1)), std(userPerformance(:,1)));
fprintf('Average Memory Usage: %.2f MB (±%.2f)\n', mean(userPerformance(:,2)), std(userPerformance(:,2)));
fprintf('Average Throughput: %.2f samples/s (±%.2f)\n', mean(userPerformance(:,3)), std(userPerformance(:,3)));

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

%% 5. Similarity Heatmap

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
colormap(parula); % safer than 'sky'
c = colorbar;
c.Label.String = 'Similarity Score';

[Xh,Yh] = meshgrid(1:numUsers, 1:numUsers);
for i = userRange_min:userRange_max
    for j = userRange_min:userRange_max
        text(j, i, labelStrings{i,j}, ...
          'HorizontalAlignment', 'center', ...
          'Color', 'black', ...
          'FontSize', 10);

        if leaveOutUsersList(i) == j
          hold on;
          plot(j, i, 'rs', 'MarkerSize', 60);
          hold off;
        end
    end
end

set(gca, 'XTick', 1:numUsers, 'XTickLabel', userRange_min:userRange_max);
set(gca, 'YTick', 1:numUsers, 'YTickLabel', userRange_min:userRange_max);
xlabel("User N's similarity score");
ylabel("User N's Model");
title('User similarity scores for each user model');
axis square;

% Save results
save('benchmark_results.mat', 'summaryTable', 'overallMetrics', 'results');
save('user_authentication_models.mat', 'models');
save('feature_analysis_results.mat', 'featureAnalysis');

%% Helper: Mutual Information for a single feature vs binary label
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
