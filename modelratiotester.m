%% modelratiotester_EER.m
% FeedForwardNet with Ratio Splitting Performance Benchmarking
% UPDATED: z-score normalization, EER-based threshold.

clear all; close all; clc;
rng(100);  % For reproducibility

% Define parameters
userRange_min = 1;
userRange_max = 10;
numUsers      = userRange_max - userRange_min + 1;

ratios = [ ...
  1/1, ...
  1/2, ...
  1/3, ...
  1/4, ...
  1/5, ...
  1/6, ...
  1/7 ...
  ];

% Neural Network parameters
dropoutRate           = 0.3;  %#ok<NASGU>
l2RegParam            = 1e-4;
performanceGoal       = 1e-5;
minGrad               = 1e-6;
earlyStoppingPatience = 10;
maxEpochs             = 500;
learningRate          = 0.01;

% Feature set pairs
featureSets = {
  {'Acc_TimeD_FreqD_FDay', 'Acc_TimeD_FreqD_MDay'},
  {'Acc_TimeD_FDay',       'Acc_TimeD_MDay'},
  {'Acc_FreqD_FDay',       'Acc_FreqD_MDay'}
  };

featureSetLabels = {'TimeD+FreqD','TimeD','FreqD'};

allResults = cell(length(featureSets), 1);

for setIdx = 1:length(featureSets)
  fprintf('\n\nProcessing feature set: %s\n', featureSets{setIdx}{1});

  userData = struct();
  for user = userRange_min:userRange_max
    userStr = sprintf('U%02d', user);

    trainFile = fullfile('dataset', [userStr '_' featureSets{setIdx}{1} '.mat']);
    testFile  = fullfile('dataset', [userStr '_' featureSets{setIdx}{2} '.mat']);

    if exist(trainFile, 'file') && exist(testFile, 'file')
      trainData = load(trainFile);
      testData  = load(testFile);

      userData(user).trainFeatures = trainData.(char(fieldnames(trainData)));
      userData(user).testFeatures  = testData.(char(fieldnames(testData)));
    else
      fprintf('Missing data files for user %d\n', user);
      userData(user).trainFeatures = [];
      userData(user).testFeatures  = [];
    end
  end

  ratioResults = cell(length(ratios), 1);

  for ratioIndex = 1:length(ratios)
    currentRatio = ratios(ratioIndex);
    fprintf('\n\n--- Benchmarking Ratio 1:%d ---\n', round(1/currentRatio));

    userMetrics = zeros(userRange_max, 14);

    for targetUser = userRange_min:userRange_max
      if isempty(userData(targetUser).trainFeatures) || isempty(userData(targetUser).testFeatures)
        warning('Skipping user %d for this ratio/feature set (no data).', targetUser);
        continue;
      end

      %% Build train set
      targetFeatures = userData(targetUser).trainFeatures;
      numTargetSamples = min(36, size(targetFeatures, 1));
      idxTarget = randperm(size(targetFeatures,1), numTargetSamples);
      targetSamples = targetFeatures(idxTarget, :);
      targetLabels  = ones(numTargetSamples, 1);

      numImposterSamplesNeeded = round(numTargetSamples * (1/currentRatio));

      impFeatures = [];
      impLabels   = [];
      samplesPerImposter = ceil(numImposterSamplesNeeded / (numUsers - 1));

      for imposterUser = userRange_min:userRange_max
        if imposterUser ~= targetUser && ~isempty(userData(imposterUser).trainFeatures)
          impData = userData(imposterUser).trainFeatures;
          nAvail  = size(impData,1);
          k       = min(samplesPerImposter, nAvail);
          idxImp  = randperm(nAvail, k);
          impFeatures = [impFeatures; impData(idxImp,:)];
          impLabels   = [impLabels; zeros(k,1)];
        end
      end

      X_train_raw = [targetSamples; impFeatures];
      y_train     = [targetLabels; impLabels];

      shuffleIdx  = randperm(size(X_train_raw,1));
      X_train_raw = X_train_raw(shuffleIdx,:);
      y_train     = y_train(shuffleIdx,:);

      % z-score normalization for train
      [X_train, mu, sigma] = zscore(X_train_raw);
      sigma(sigma == 0) = 1;

      %% Build test set
      testFeaturesPos = userData(targetUser).testFeatures;
      testLabelsPos   = ones(size(testFeaturesPos, 1), 1);

      testFeaturesNeg = [];
      testLabelsNeg   = [];
      samplesPerImposterTest = ceil(size(testFeaturesPos, 1) / (numUsers - 1));

      for imposterUser = userRange_min:userRange_max
        if imposterUser ~= targetUser && ~isempty(userData(imposterUser).testFeatures)
          impTestData = userData(imposterUser).testFeatures;
          nAvail = size(impTestData,1);
          k      = min(samplesPerImposterTest, nAvail);
          idxImp = randperm(nAvail, k);
          testFeaturesNeg = [testFeaturesNeg; impTestData(idxImp,:)];
          testLabelsNeg   = [testLabelsNeg; zeros(k,1)];
        end
      end

      X_test_raw = [testFeaturesPos; testFeaturesNeg];
      y_test     = [testLabelsPos; testLabelsNeg];

      % Apply train normalization
      X_test = (X_test_raw - mu) ./ sigma;

      %% Neural Network setup and training
      net = feedforwardnet(64, 'trainscg'); % slightly smaller
      net.performFcn = 'crossentropy';
      net.layers{1}.transferFcn  = 'tansig';
      net.layers{end}.transferFcn = 'tansig';

      net.trainParam.epochs   = maxEpochs;
      net.trainParam.goal     = performanceGoal;
      net.trainParam.min_grad = minGrad;
      net.performParam.regularization = l2RegParam;
      net.trainParam.max_fail = earlyStoppingPatience;
      net.trainParam.lr       = learningRate;
      net.trainParam.showWindow      = false;
      net.trainParam.showCommandLine = false;

      net = train(net, X_train', y_train');

      %% Evaluation with EER threshold
      scores = net(X_test')';
      yTrueBin = (y_test == 1);

      [FPR, TPR, Thr, auc] = perfcurve(yTrueBin, scores, true);
      FAR = FPR;
      FRR = 1 - TPR;

      [~, eerIdx] = min(abs(FAR - FRR));
      EER = (FAR(eerIdx) + FRR(eerIdx))/2;
      bestThr = Thr(eerIdx);

      yPredBin = double(scores >= bestThr);

      tp = sum(yPredBin==1 & yTrueBin==1);
      fp = sum(yPredBin==1 & yTrueBin==0);
      fn = sum(yPredBin==0 & yTrueBin==1);
      tn = sum(yPredBin==0 & yTrueBin==0);

      accuracy   = sum(yPredBin == yTrueBin)/numel(yTrueBin);
      precision  = tp/(tp+fp+eps);
      recall     = tp/(tp+fn+eps);
      specificity= tn/(tn+fp+eps);
      f1_score   = 2*(precision*recall)/(precision+recall+eps);
      mcc        = ((tp*tn)-(fp*fn))/sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)+eps);

      userMetrics(targetUser, :) = [accuracy, precision, recall, specificity, ...
        f1_score, mcc, FAR(eerIdx)*100, FRR(eerIdx)*100, EER*100, auc, ...
        size(X_train, 1), size(targetSamples, 1), size(impFeatures, 1), ...
        size(X_test, 1)];
    end

    avgMetrics = mean(userMetrics, 1);

    ratioResults{ratioIndex} = struct( ...
      'Ratio', sprintf('1:%d', round(1/currentRatio)), ...
      'AvgAccuracy',         avgMetrics(1)*100, ...
      'AvgPrecision',        avgMetrics(2)*100, ...
      'AvgRecall',           avgMetrics(3)*100, ...
      'AvgSpecificity',      avgMetrics(4)*100, ...
      'AvgF1Score',          avgMetrics(5)*100, ...
      'AvgMCC',              avgMetrics(6), ...
      'AvgFAR',              avgMetrics(7), ...
      'AvgFRR',              avgMetrics(8), ...
      'AvgEER',              avgMetrics(9), ...
      'AvgAUC',              avgMetrics(10), ...
      'AvgTrainingSetSize',  avgMetrics(11), ...
      'AvgTargetSamples',    avgMetrics(12), ...
      'AvgImposterSamples',  avgMetrics(13), ...
      'AvgTestSetSize',      avgMetrics(14) ...
      );

    fprintf('\nResults for Ratio 1:%d\n', round(1/currentRatio));
    disp(ratioResults{ratioIndex});
  end

  allResults{setIdx} = ratioResults;

  fprintf('\n\nResults for Feature Set: %s and %s\n', ...
    featureSets{setIdx}{1}, featureSets{setIdx}{2});
  resultsTable = table( ...
    cellfun(@(x) x.Ratio, ratioResults, 'UniformOutput', false), ...
    cellfun(@(x) x.AvgAccuracy,  ratioResults), ...
    cellfun(@(x) x.AvgPrecision, ratioResults), ...
    cellfun(@(x) x.AvgRecall,    ratioResults), ...
    cellfun(@(x) x.AvgSpecificity, ratioResults), ...
    cellfun(@(x) x.AvgF1Score,   ratioResults), ...
    cellfun(@(x) x.AvgMCC,       ratioResults), ...
    cellfun(@(x) x.AvgFAR,       ratioResults), ...
    cellfun(@(x) x.AvgFRR,       ratioResults), ...
    cellfun(@(x) x.AvgEER,       ratioResults), ...
    cellfun(@(x) x.AvgAUC,       ratioResults), ...
    cellfun(@(x) x.AvgTrainingSetSize, ratioResults), ...
    cellfun(@(x) x.AvgTargetSamples,   ratioResults), ...
    cellfun(@(x) x.AvgImposterSamples, ratioResults), ...
    cellfun(@(x) x.AvgTestSetSize,     ratioResults), ...
    'VariableNames', { ...
    'Ratio', 'Accuracy', 'Precision', 'Recall', 'Specificity', ...
    'F1Score', 'MCC', 'FAR', 'FRR', 'EER', 'AUC', ...
    'TrainingSetSize', 'TargetSamples', 'ImposterSamples', 'TestSetSize'});

  disp(resultsTable);
end

% Plot curves across ratios
ratioNums = 1./ratios;

figure('Position', [100 100 1200 800]);
colors = {'b-o', 'r-s', 'g-^'};
legendLabels = featureSetLabels;

metricsNames = {'Accuracy', 'MCC', 'FAR', 'FRR'};
titles       = {'Accuracy vs Ratio', ...
                'MCC vs Ratio', ...
                'FAR vs Ratio', ...
                'FRR vs Ratio'};
ylabels      = {'Accuracy (%)','MCC','FAR (%)','FRR (%)'};

for i = 1:4
  subplot(3,2,i);
  hold on;
  for setIdx = 1:length(featureSets)
    vals = cellfun(@(x) x.(sprintf('Avg%s', metricsNames{i})), allResults{setIdx});
    if any(strcmp(metricsNames{i},{'Accuracy','FAR','FRR'}))
        valsPlot = vals; % already %
    else
        valsPlot = vals;
    end
    plot(ratioNums, valsPlot, colors{setIdx}, 'LineWidth', 1.5);
  end
  title(titles{i});
  xlabel('Ratio (1:N)');
  ylabel(ylabels{i});
  grid on;
  legend(legendLabels, 'Location', 'best');
  hold off;
end

subplot('Position', [0.125, 0.1, 0.8, 0.2]);
hold on;
for setIdx = 1:length(featureSets)
  vals = cellfun(@(x) x.AvgEER, allResults{setIdx});
  plot(ratioNums, vals, colors{setIdx}, 'LineWidth', 1.5);
end
title('Equal Error Rate (EER) vs Ratio');
xlabel('Ratio (1:N)');
ylabel('EER (%)');
grid on;
legend(legendLabels, 'Location', 'best');
hold off;

sgtitle('Performance Metrics Across Different Ratio Splits and Feature Sets (EER thresholds)');

save('ratio_splitting_performance_all_features_EER.mat', 'allResults');
