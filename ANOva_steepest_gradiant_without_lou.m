%% ANOVA + MI + Steepest Gradient Feature Selection (No LOU, CLEAN)
% Complete User Authentication using MLP Neural Network - Binary Classification

clear all; close all; clc;
rng(100);  % For reproducibility

% Define script params
userRange_min = 1;
userRange_max = 10;
numUsers      = userRange_max - userRange_min + 1;

% Overfitting prevention parameters
TrainTargetImposterRatio = 1/5;  % Fixed ratio 1:5
dropoutRate              = 0.3;  % (stored in userdata only)
l2RegParam               = 1e-4; % L2 regularization parameter
performanceGoal          = 1e-5; % Performance goal for training
minGrad                  = 1e-6; % Minimum gradient for training
earlyStoppingPatience    = 10;   % Patience for early stopping
maxEpochs                = 500;  % Maximum number of training epochs
learningRate             = 0.01; % Learning rate

epsVal = 1e-12; % numerical safety

%% 1. Data Loading and Preprocessing
filePatternsTrain = 'Acc_TimeD_FreqD_FDay';
filePatternsTest  = 'Acc_TimeD_FreqD_MDay';

fprintf('Loading data for each user...\n');

% Pre-allocate userData
userData = struct('trainFeatures', [], 'testFeatures', []);
userData = repmat(userData, 1, userRange_max);

for user = userRange_min:userRange_max
    userStr = sprintf('U%02d', user);

    trainBase = [userStr '_' filePatternsTrain '.mat'];
    testBase  = [userStr '_' filePatternsTest  '.mat'];

    % Try current folder first
    trainFile = trainBase;
    testFile  = testBase;

    if ~exist(trainFile, 'file') || ~exist(testFile, 'file')
        % Then try inside "dataset" subfolder
        trainFile = fullfile('dataset', trainBase);
        testFile  = fullfile('dataset', testBase);
    end

    if exist(trainFile, 'file') && exist(testFile, 'file')
        trainData = load(trainFile);
        testData  = load(testFile);

        userData(user).trainFeatures = trainData.(char(fieldnames(trainData)));
        userData(user).testFeatures  = testData.(char(fieldnames(testData)));

        [r,c] = size(userData(user).trainFeatures);
        fprintf('User %d: %d train samples x %d features\n', user, r, c);
    else
        fprintf('Missing data files for user %d\n', user);
        userData(user).trainFeatures = [];
        userData(user).testFeatures  = [];
    end
end

%% 2. Feature Selection (ANOVA + MI + Gradient, no LOU)

anovaThreshold   = 0.05;  % Threshold for ANOVA p-values
topFeaturePercent = 0.75; % Top 75% features to select

selectedFeatures = cell(numUsers, 1);
featureAnalysis  = struct();

for targetUser = userRange_min:userRange_max
    if isempty(userData(targetUser).trainFeatures)
        fprintf('Skipping feature selection for user %d (no train data)\n', targetUser);
        continue;
    end

    % Prepare data for feature selection
    X = userData(targetUser).trainFeatures;
    y = ones(size(X, 1), 1);

    for imposterUser = userRange_min:userRange_max
        if imposterUser ~= targetUser && ~isempty(userData(imposterUser).trainFeatures)
            X = [X; userData(imposterUser).trainFeatures];
            y = [y; zeros(size(userData(imposterUser).trainFeatures, 1), 1)];
        end
    end

    numFeatures = size(X, 2);
    fprintf('\n=== Feature Selection for User %d (%d features) ===\n', targetUser, numFeatures);

    % --- ANOVA Feature Selection ---
    pValues = zeros(1, numFeatures);
    for i = 1:numFeatures
        pValues(i) = anova1(X(:, i), y, 'off');
    end
    anovaSelected = find(pValues < anovaThreshold);

    % --- Mutual Information Feature Selection ---
    miScores = zeros(1, numFeatures);
    for i = 1:numFeatures
        miScores(i) = calculate_mutual_information(X(:, i), y);
    end
    [~, miRanking] = sort(miScores, 'descend');
    miSelected = miRanking(1:round(topFeaturePercent*numFeatures));

    % --- Steepest Gradient Feature Selection ---
    net_fs = feedforwardnet(10, 'trainscg');
    net_fs.trainParam.showWindow      = false;
    net_fs.trainParam.showCommandLine = false;
    net_fs = train(net_fs, X', y');
    gradients = abs(net_fs.IW{1});
    [~, sgRanking] = sort(mean(gradients, 1), 'descend');
    sgSelected = sgRanking(1:round(topFeaturePercent*numFeatures));

    % --- Feature Scores for Visualization ---
    figure('Name', sprintf('Feature Analysis - User %d', targetUser));

    subplot(2,2,1);
    featureScores = zeros(numFeatures, 3);

    % ANOVA: lower p-value = more important
    featureScores(:,1) = 1 - normalize(reshape(pValues, [], 1), 'range');

    % MI
    featureScores(:,2) = normalize(reshape(miScores, [], 1), 'range');

    % Gradient scores
    meanGradients = mean(gradients, 1)';
    if length(meanGradients) ~= numFeatures
        meanGradients = interp1(1:length(meanGradients), meanGradients, ...
                                linspace(1, length(meanGradients), numFeatures));
    end
    featureScores(:,3) = normalize(meanGradients, 'range');

    bar(featureScores, 'stacked');
    title(sprintf('Feature Importance by Method (%d features)', numFeatures));
    legend('ANOVA', 'MI', 'Gradient');
    xlabel('Feature Index');
    ylabel('Normalized Importance');

    % Weighted combined score
    weights        = [0.4, 0.3, 0.3];
    combinedScores = featureScores * weights';
    [~, sortedIdx] = sort(combinedScores, 'descend');
    selectedFeatureIdx = sortedIdx(1:round(topFeaturePercent*numFeatures));

    % Correlation matrix for selected features
    subplot(2,2,2);
    correlationMatrix = corr(X(:, selectedFeatureIdx));
    imagesc(correlationMatrix);
    colormap(jet);
    colorbar;
    title(sprintf('Feature Correlation Matrix\n(%d features)', length(selectedFeatureIdx)));

    % Box plots for top features (visual only; not perfect grouping)
    subplot(2,2,3);
    topN = min(5, length(selectedFeatureIdx));
    topFeatures = selectedFeatureIdx(1:topN);

    boxData = [];
    groupLabels = {};
    for i = 1:topN
        featureValues = X(:, topFeatures(i));
        boxData = [boxData; featureValues]; %#ok<AGROW>
        groupLabels = [groupLabels; repmat({sprintf('Feature %d', i)}, length(featureValues), 1)]; %#ok<AGROW>
    end

    boxplot(boxData, groupLabels);
    title(sprintf('Top %d Features Distribution', topN));
    xlabel('Feature Index');
    ylabel('Feature Value');
    grid on;

    % Method contribution pie
    subplot(2,2,4);
    methodContribution = sum(featureScores, 1);
    if all(isfinite(methodContribution))
        pie(methodContribution);
        title('Feature Selection Method Contribution');
        legend({'ANOVA', 'MI', 'Gradient'}, 'Location', 'eastoutside');
    else
        warning('Non-finite values in method contribution; skipping pie chart.');
    end

    % Additional visualization
    figure('Name', sprintf('Feature Importance Trends - User %d', targetUser));

    subplot(2,1,1);
    combinedScoresMean = mean(featureScores, 2);
    [sortedScores2, sortedIdx2] = sort(combinedScoresMean, 'descend');
    bar(sortedScores2(1:min(20,end)));
    title('Top 20 Features by Combined Importance');
    xlabel('Feature Rank');
    ylabel('Importance Score');
    grid on;

    subplot(2,1,2);
    topK = min(20, length(sortedIdx2));
    topFeatureScores = featureScores(sortedIdx2(1:topK), :);
    bar(topFeatureScores, 'grouped');
    title('Method Contributions for Top Features');
    xlabel('Feature Rank');
    ylabel('Score');
    legend('ANOVA','MI','Gradient');
    grid on;

    % Store analysis
    featureAnalysis(targetUser).scores           = featureScores;
    featureAnalysis(targetUser).combinedScores   = combinedScoresMean;
    featureAnalysis(targetUser).selectedFeatures = selectedFeatureIdx;
    featureAnalysis(targetUser).correlationMatrix = correlationMatrix;

    fprintf('\nFeature Selection Summary for User %d:\n', targetUser);
    fprintf('Top 5 features: %s\n', mat2str(selectedFeatureIdx(1:min(5,end))));
    fprintf('Average correlation: %.4f\n', ...
        mean(abs(correlationMatrix(triu(true(size(correlationMatrix)),1)))));
    fprintf('ANOVA selected: %d, MI selected: %d, SG selected: %d, Combined: %d\n', ...
        length(anovaSelected), length(miSelected), length(sgSelected), length(selectedFeatureIdx));

    selectedFeatures{targetUser} = selectedFeatureIdx;
end

%% 3. Model Training & Evaluation (No LOU, using selected features)

models           = cell(numUsers, 1);
userMetrics      = zeros(numUsers, 15);
userPerformance  = zeros(numUsers, 3); % [time, mem, throughput]
userSimilarityData = cell(3, numUsers, numUsers);

for targetUser = userRange_min:userRange_max
    if isempty(userData(targetUser).trainFeatures) || isempty(selectedFeatures{targetUser})
        fprintf('Skipping training for user %d (no data or no features).\n', targetUser);
        continue;
    end

    selectedIdx = selectedFeatures{targetUser};

    % --- Training set ---
    trainTargetSampleCount   = size(userData(targetUser).trainFeatures, 1);
    trainImposterSampleCount = trainTargetSampleCount*(1/TrainTargetImposterRatio);
    trainSamplesPerImposter  = floor(trainImposterSampleCount/(numUsers-1));

    XTrain = userData(targetUser).trainFeatures(:, selectedIdx);
    yTrain = ones(trainTargetSampleCount, 1);

    trainImposterFeatures = [];
    trainImposterLabels   = [];

    for imposterUser = userRange_min:userRange_max
        if imposterUser == targetUser, continue; end
        if isempty(userData(imposterUser).trainFeatures), continue; end

        nAvail = size(userData(imposterUser).trainFeatures, 1);
        nTake  = min(trainSamplesPerImposter, nAvail);
        if nTake <= 0, continue; end

        idx = randperm(nAvail, nTake);
        trainImposterFeatures = [trainImposterFeatures; ...
            userData(imposterUser).trainFeatures(idx, selectedIdx)];
        trainImposterLabels = [trainImposterLabels; zeros(nTake, 1)];
    end

    XTrain = [XTrain; trainImposterFeatures];
    yTrain = [yTrain; trainImposterLabels];

    assert(sum(yTrain == 1) == trainTargetSampleCount);
    actualImposterSamples = sum(yTrain == 0);
    fprintf('\nTarget user %d: requested imposters = %d, actual = %d\n', ...
        targetUser, trainImposterSampleCount, actualImposterSamples);
    trainImposterSampleCount = actualImposterSamples;

    % --- Test set ---
    testTargetSampleCount   = size(userData(targetUser).testFeatures, 1);
    testImposterSampleCount = 324;
    testSamplesPerImposter  = floor(testImposterSampleCount/(numUsers-1));

    XTest = userData(targetUser).testFeatures(:, selectedIdx);
    yTest = ones(testTargetSampleCount, 1);

    testImposterFeatures = [];
    testImposterLabels   = [];
    testUserLabels       = ones(testTargetSampleCount, 1) * targetUser;

    for imposterUser = userRange_min:userRange_max
        if imposterUser == targetUser, continue; end
        if isempty(userData(imposterUser).testFeatures), continue; end

        nAvail = size(userData(imposterUser).testFeatures, 1);
        nTake  = min(testSamplesPerImposter, nAvail);
        if nTake <= 0, continue; end

        idx = randperm(nAvail, nTake);
        testImposterFeatures = [testImposterFeatures; ...
            userData(imposterUser).testFeatures(idx, selectedIdx)];
        testImposterLabels = [testImposterLabels; zeros(nTake, 1)];
        testUserLabels = [testUserLabels; ones(nTake, 1) * imposterUser];
    end

    XTest = [XTest; testImposterFeatures];
    yTest = [yTest; testImposterLabels];

    assert(sum(yTest == 1) == testTargetSampleCount);
    assert(sum(yTest == 0) == testImposterSampleCount);

    % --- Network definition ---
    net = feedforwardnet(57, 'trainscg');
    net.userdata.note                     = "FFNN with ANOVA+MI+Gradient (no LOU)";
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
    net.layers{1}.transferFcn   = 'logsig';
    net.layers{end}.transferFcn = 'tansig';

    net.trainParam.epochs   = maxEpochs;
    net.trainParam.goal     = performanceGoal;
    net.trainParam.min_grad = minGrad;
    net.performParam.regularization = l2RegParam;
    net.trainParam.max_fail = earlyStoppingPatience;
    net.trainParam.lr       = learningRate;

    net.trainParam.showWindow      = false;
    net.trainParam.showCommandLine = false;

    % --- Train ---
    tic;
    [net, tr] = train(net, XTrain', yTrain');
    trainTime = toc;
    models{targetUser} = net;

    modelInfo   = whos('net');
    memoryUsage = modelInfo.bytes / (1024^2);

    % --- Evaluate ---
    tic;
    rawPredictions = net(XTest')';   % continuous scores
    inferenceTime  = toc;
    throughput     = size(XTest, 1) / inferenceTime;

    yPred = double(rawPredictions > 0.5);

    tp = sum(yPred == 1 & yTest == 1);
    tn = sum(yPred == 0 & yTest == 0);
    fp = sum(yPred == 1 & yTest == 0);
    fn = sum(yPred == 0 & yTest == 1);

    genuinePrecision  = tp / max(tp + fp, epsVal);
    impostorPrecision = tn / max(tn + fn, epsVal);

    overallPrecision = (genuinePrecision * sum(yTest == 1) + ...
                        impostorPrecision * sum(yTest == 0)) / max(length(yTest),1);

    recall      = tp / max(tp + fn, epsVal);
    specificity = tn / max(tn + fp, epsVal);
    accuracy    = (tp + tn) / max(tp+tn+fp+fn, 1);

    f1_score = 2 * overallPrecision * recall / max(overallPrecision + recall, epsVal);

    fpr = fp / max(fp + tn, epsVal);
    fnr = fn / max(fn + tp, epsVal);
    eer = (fnr + fpr) / 2;

    mcc_num = (tp * tn) - (fp * fn);
    mcc_den = sqrt(max((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn), epsVal));
    mcc     = mcc_num / max(mcc_den, epsVal);

    % ROC & AUC from continuous scores
    [FAR, TPR, ~, AUC] = perfcurve(yTest, rawPredictions, 1);
    FRR = 1 - TPR;
    [~, eerIdx] = min(abs(FAR - FRR));
    EER = (FAR(eerIdx) + FRR(eerIdx))/2;

    % Store performance
    userPerformance(targetUser, :) = [trainTime + inferenceTime, memoryUsage, throughput];

    % Similarities
    modelUserSimilarities = [testUserLabels, rawPredictions];

    similarity_means          = zeros(1, numUsers);
    similarity_mids           = zeros(1, numUsers);
    similarity_mid_variations = zeros(1, numUsers);

    for u = userRange_min:userRange_max
        idx_u = find(modelUserSimilarities(:,1) == u);
        if ~isempty(idx_u)
            vals = modelUserSimilarities(idx_u, 2);
            mn   = min(vals);
            mx   = max(vals);
            similarity_means(1,u) = mean(vals);
            similarity_mids(1,u)  = (mx + mn)/2;
            similarity_mid_variations(1,u) = mx - similarity_mids(1,u);
        end
    end

    userSimilarityData(1, targetUser, :) = num2cell(similarity_means);
    userSimilarityData(2, targetUser, :) = num2cell(similarity_mids);
    userSimilarityData(3, targetUser, :) = num2cell(similarity_mid_variations);

    % Store metrics: [acc, precision, recall, spec, f1, mcc, FAR, FRR, EER, AUC, OverallPrecision, trainSet, trainTarget, trainImposter, testSet]
    userMetrics(targetUser, :) = [ ...
        accuracy, overallPrecision, recall, specificity, ...
        f1_score, mcc, fpr*100, fnr*100, eer*100, AUC, overallPrecision, ...
        size(XTrain,1), trainTargetSampleCount, trainImposterSampleCount, ...
        size(XTest,1)];

    fprintf('\nUser %d Results:\n', targetUser);
    fprintf('Accuracy: %.2f%%\n', accuracy*100);
    fprintf('Overall Precision: %.2f%%\n', overallPrecision*100);
    fprintf('Recall: %.2f%%\n', recall*100);
    fprintf('Specificity: %.2f%%\n', specificity*100);
    fprintf('F1-Score: %.2f%%\n', f1_score*100);
    fprintf('MCC: %.4f\n', mcc);
    fprintf('FAR: %.2f%%, FRR: %.2f%%, EER: %.2f%%\n', fpr*100, fnr*100, eer*100);
    fprintf('AUC: %.4f\n', AUC);
    fprintf('Total Time: %.4f s, Mem: %.2f MB, Throughput: %.2f samples/s\n', ...
        userPerformance(targetUser,1), userPerformance(targetUser,2), userPerformance(targetUser,3));

    % Confusion matrix (Neural Network Toolbox plot)
    figure;
    plotconfusion(yTest', yPred');
    title(sprintf('Confusion Matrix - User %d', targetUser));

    % ROC curve
    figure;
    plot(FAR, TPR);
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    title(sprintf('ROC Curve - User %d (AUC = %.3f)', targetUser, AUC));
    grid on;
end

%% 4. Averages & Summary

avgMetrics      = mean(userMetrics, 1, 'omitnan');
avgPerformance  = mean(userPerformance, 1, 'omitnan');

results = struct( ...
  'Ratio',                  '1:5', ...
  'AvgAccuracy',            avgMetrics(1)*100, ...
  'AvgOverallPrecision',    avgMetrics(11)*100, ...
  'AvgRecall',              avgMetrics(3)*100, ...
  'AvgSpecificity',         avgMetrics(4)*100, ...
  'AvgF1Score',             avgMetrics(5)*100, ...
  'AvgMCC',                 avgMetrics(6), ...
  'AvgFAR',                 avgMetrics(7), ...
  'AvgFRR',                 avgMetrics(8), ...
  'AvgEER',                 avgMetrics(9), ...
  'AvgAUC',                 avgMetrics(10), ...
  'AvgTrainingSetSize',     avgMetrics(12), ...
  'AvgTrainTargetSamples',  avgMetrics(13), ...
  'AvgTrainImposterSamples',avgMetrics(14), ...
  'AvgTestSetSize',         avgMetrics(15), ...
  'AvgTotalTime',           avgPerformance(1), ...
  'AvgMemoryUsage',         avgPerformance(2), ...
  'AvgThroughput',          avgPerformance(3));

fprintf('\n==== Neural Network Architecture ====\n');
fprintf('Input Layer: %d neurons (selected features)\n', size(XTrain, 2));
fprintf('Hidden Layer: 57 neurons (logsig)\n');
fprintf('Output Layer: 1 neuron (tansig)\n');
fprintf('Training Algorithm: trainscg\n');
fprintf('Performance Function: crossentropy\n');
fprintf('L2 Regularization: %e\n', l2RegParam);
fprintf('Max Epochs: %d\n', maxEpochs);

fprintf('\n==== Performance Benchmarks ====\n');
fprintf('Average Training+Inference Time: %.4f s (±%.4f)\n', ...
    mean(userPerformance(:,1),'omitnan'), std(userPerformance(:,1),'omitnan'));
fprintf('Average Memory Usage: %.2f MB (±%.2f)\n', ...
    mean(userPerformance(:,2),'omitnan'), std(userPerformance(:,2),'omitnan'));
fprintf('Average Throughput: %.2f samples/s (±%.2f)\n', ...
    mean(userPerformance(:,3),'omitnan'), std(userPerformance(:,3),'omitnan'));

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
  'Accuracy', 'OverallPrecision', 'Recall', 'Specificity', 'F1_Score', ...
  'MCC', 'FAR', 'FRR', 'EER', 'AUC'});

overallMetrics = table( ...
  mean(userPerformance(:,1),'omitnan'), ...
  mean(userPerformance(:,2),'omitnan'), ...
  mean(userPerformance(:,3),'omitnan'), ...
  mean(userMetrics(:,1)*100,'omitnan'), ...
  mean(userMetrics(:,2)*100,'omitnan'), ...
  mean(userMetrics(:,3)*100,'omitnan'), ...
  mean(userMetrics(:,4)*100,'omitnan'), ...
  mean(userMetrics(:,5)*100,'omitnan'), ...
  mean(userMetrics(:,6),'omitnan'), ...
  mean(userMetrics(:,7),'omitnan'), ...
  mean(userMetrics(:,8),'omitnan'), ...
  mean(userMetrics(:,9),'omitnan'), ...
  mean(userMetrics(:,10),'omitnan'), ...
  'VariableNames', { ...
  'Avg_TotalTime_sec', 'Avg_MemoryUsage_MB', 'Avg_Throughput_samples_per_sec', ...
  'Avg_Accuracy', 'Avg_OverallPrecision', 'Avg_Recall', 'Avg_Specificity', 'Avg_F1_Score', ...
  'Avg_MCC', 'Avg_FAR', 'Avg_FRR', 'Avg_EER', 'Avg_AUC'});

fprintf('\n==== Summary Table ====\n');
disp(summaryTable);
disp('Overall Metrics:');
disp(overallMetrics);

%% Similarity heatmap
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
colormap(parula);
c = colorbar;
c.Label.String = 'Similarity Score';

[Xh, Yh] = meshgrid(1:numUsers, 1:numUsers);
for i = userRange_min:userRange_max
  for j = userRange_min:userRange_max
    text(j, i, labelStrings{i,j}, ...
      'HorizontalAlignment', 'center', ...
      'Color', 'black', ...
      'FontSize', 10);
  end
end

set(gca, 'XTick', 1:numUsers, 'XTickLabel', userRange_min:userRange_max);
set(gca, 'YTick', 1:numUsers, 'YTickLabel', userRange_min:userRange_max);
xlabel("User N's similarity score");
ylabel("User N's Model");
title('User similarity scores for each user model (ANOVA+MI+Gradient, no LOU)');
axis square;

% Save the results
save('benchmark_results_anova_mi_gradient_noLOU.mat', 'summaryTable', 'overallMetrics', 'results');
save('user_authentication_models_anova_mi_gradient_noLOU.mat', 'models');
save('feature_analysis_results_anova_mi_gradient_noLOU.mat', 'featureAnalysis');

%% Helper: Mutual Information
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
                mi = mi + joint_p(i,j) * log2(joint_p(i,j) / (p_x(i) * p_y(j) + eps) + eps);
            end
        end
    end
end
