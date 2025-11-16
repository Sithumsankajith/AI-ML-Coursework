% Multi-user classification with Frequency / Time / Time-Frequency features
% - Loads Uxx_Acc_*.mat from folder "dataset"
% - Computes inter- / intra-user variance and suitability
% - Trains 3 feedforward NNs (Freq, Time, Time-Freq)
% - Reports Accuracy, Precision, Recall, F1 and plots confusion/performance

clear; clc; close all;

% Disable automatic figure export callbacks (safer for long runs)
set(0, 'DefaultFigureCreateFcn', '');
set(0, 'DefaultFigureDeleteFcn', '');


%% Add dataset path
addpath('dataset');   % <-- make sure your .mat files are in this folder

%% Parameters
users    = 1:10;  % User IDs from U01 to U10
numUsers = numel(users);

% File types you have in your dataset
fileTypes = { ...
    'Acc_FreqD_FDay', ...          % 1
    'Acc_TimeD_FDay', ...          % 2
    'Acc_TimeD_FreqD_FDay', ...    % 3
    'Acc_FreqD_MDay', ...          % 4
    'Acc_TimeD_MDay', ...          % 5
    'Acc_TimeD_FreqD_MDay'};       % 6

% Indices for feature set groups
freqIdx      = [1, 4];  % Frequency-only (FDay + MDay)
timeIdx      = [2, 5];  % Time-only      (FDay + MDay)
timeFreqIdx  = [3, 6];  % Time+Freq      (FDay + MDay)

%% Load and Transpose Data
allData = cell(numUsers, numel(fileTypes));

fprintf('Loading feature matrices from folder "dataset"...\n');

for ui = 1:numUsers
    u = users(ui);
    for f = 1:numel(fileTypes)

        fileName = sprintf('U%02d_%s.mat', u, fileTypes{f});
        fullPath = fullfile('dataset', fileName);

        if ~exist(fullPath, 'file')
            error('Missing file: %s (expected in "dataset" folder)', fileName);
        end

        % Load .mat (we do not assume a fixed variable name)
        tmp    = load(fullPath);
        fields = fieldnames(tmp);

        % Take the first variable inside the .mat as the feature matrix
        featMat = tmp.(fields{1});   % [Samples x Features] in your coursework

        % Transpose: we want [Features x Samples] for NN and later code
        featMat = featMat';          % [Features x Samples]

        allData{ui, f} = featMat;

        fprintf('Loaded %s -> [%d features x %d samples]\n', ...
            fileName, size(featMat,1), size(featMat,2));
    end
end

%% Analyze Inter-User and Intra-User Variance
interUserVariance  = zeros(numel(fileTypes), 1);   % one per feature type
intraUserVariance  = cell(numel(fileTypes), 1);    % cell: [NF x users] per type

for f = 1:numel(fileTypes)
    combinedFeatures = [];
    % assume all users have same number of features for this type
    NF = size(allData{1,f}, 1);
    intraVar = zeros(NF, numUsers);   % [NF x users]

    for ui = 1:numUsers
        userFeatures   = allData{ui, f};  % [NF x NS]
        combinedFeatures = [combinedFeatures, userFeatures];  %#ok<AGROW>
        % Intra-user variance per feature (over samples)
        intraVar(:, ui) = var(userFeatures, 0, 2);
    end

    % Inter-user variance: variance of feature means across all samples/users
    featureMeans         = mean(combinedFeatures, 2);  % [NF x 1]
    interUserVariance(f) = var(featureMeans);

    intraUserVariance{f} = intraVar;
end

%% Evaluate Classification Suitability (Inter / Intra ratio)
classificationSuitability = zeros(numel(fileTypes), 1);

fprintf('\nClassification Suitability Analysis:\n');
for f = 1:numel(fileTypes)
    avgIntraVar = mean(intraUserVariance{f}, 2);     % [NF x 1]
    avgInterVar = interUserVariance(f);

    classificationSuitability(f) = avgInterVar / mean(avgIntraVar);

    fprintf('Feature Type: %s\n', fileTypes{f});
    fprintf('  Inter-User Variance          : %.4f\n', interUserVariance(f));
    fprintf('  Average Intra-User Variance : %.4f\n', mean(avgIntraVar));
    fprintf('  Suitability Metric (Inter/Intra): %.4f\n\n', classificationSuitability(f));
end

%% Visualize Suitability Metric
figure;
bar(classificationSuitability);
xticks(1:numel(fileTypes));
xticklabels(fileTypes);
xtickangle(30);
xlabel('Feature Type');
ylabel('Suitability Metric (Inter / Intra Variance)');
title('Classification Suitability for Each Feature Type');
grid on;

%% Save variance and suitability results
save('Classification_Suitability_Results.mat', ...
     'interUserVariance', 'intraUserVariance', 'classificationSuitability');

%% Aggregate Data per Feature Group (multi-class classification: 10 users)

% 1) Frequency-only data
freqDataAll = [];
freqLabels  = [];   % numeric class label 1..10

for ui = 1:numUsers
    u = users(ui);
    for idx = freqIdx
        curData = allData{ui, idx};    % [NF x NS]
        freqDataAll = [freqDataAll, curData];           %#ok<AGROW>
        freqLabels  = [freqLabels,  repmat(u, 1, size(curData,2))]; %#ok<AGROW>
    end
end

% 2) Time-only data
timeDataAll = [];
timeLabels  = [];

for ui = 1:numUsers
    u = users(ui);
    for idx = timeIdx
        curData = allData{ui, idx};   % [NF x NS]
        timeDataAll = [timeDataAll, curData];           %#ok<AGROW>
        timeLabels  = [timeLabels,  repmat(u, 1, size(curData,2))]; %#ok<AGROW>
    end
end

% 3) Time-frequency data
timeFreqDataAll = [];
timeFreqLabels  = [];

for ui = 1:numUsers
    u = users(ui);
    for idx = timeFreqIdx
        curData = allData{ui, idx};   % [NF x NS]
        timeFreqDataAll = [timeFreqDataAll, curData];         %#ok<AGROW>
        timeFreqLabels  = [timeFreqLabels,  repmat(u, 1, size(curData,2))]; %#ok<AGROW>
    end
end

%% Split Each Dataset into Train / Val / Test
train_ratio = 0.7;
val_ratio   = 0.15;   % remaining 0.15 used as test

[X_freq_train, Y_freq_train, X_freq_val, Y_freq_val, X_freq_test, Y_freq_test] = ...
    split_dataset(freqDataAll, freqLabels, train_ratio, val_ratio, numUsers);

[X_time_train, Y_time_train, X_time_val, Y_time_val, X_time_test, Y_time_test] = ...
    split_dataset(timeDataAll, timeLabels, train_ratio, val_ratio, numUsers);

[X_timeFreq_train, Y_timeFreq_train, X_timeFreq_val, Y_timeFreq_val, ...
    X_timeFreq_test, Y_timeFreq_test] = ...
    split_dataset(timeFreqDataAll, timeFreqLabels, train_ratio, val_ratio, numUsers);

%% Feature Selection via MRMR (fscmrmr) on the Training Sets

% Choose how many features to keep (tune if needed)
k_freq     = min(40, size(X_freq_train,1));       % <= #features
k_time     = min(80, size(X_time_train,1));
k_timeFreq = min(120, size(X_timeFreq_train,1));

% Frequency-only feature selection
freqFeatureRanks     = fscmrmr(X_freq_train', vec2ind(Y_freq_train)'); % [NF x 1] ranking
freqSelectedFeatures = freqFeatureRanks(1:k_freq);
X_freq_train         = X_freq_train(freqSelectedFeatures, :);
X_freq_val           = X_freq_val(freqSelectedFeatures, :);
X_freq_test          = X_freq_test(freqSelectedFeatures, :);

% Time-only feature selection
timeFeatureRanks     = fscmrmr(X_time_train', vec2ind(Y_time_train)');
timeSelectedFeatures = timeFeatureRanks(1:k_time);
X_time_train         = X_time_train(timeSelectedFeatures, :);
X_time_val           = X_time_val(timeSelectedFeatures, :);
X_time_test          = X_time_test(timeSelectedFeatures, :);

% Time-Frequency feature selection
timeFreqFeatureRanks     = fscmrmr(X_timeFreq_train', vec2ind(Y_timeFreq_train)');
timeFreqSelectedFeatures = timeFreqFeatureRanks(1:k_timeFreq);
X_timeFreq_train         = X_timeFreq_train(timeFreqSelectedFeatures, :);
X_timeFreq_val           = X_timeFreq_val(timeFreqSelectedFeatures, :);
X_timeFreq_test          = X_timeFreq_test(timeFreqSelectedFeatures, :);

fprintf('\nFeature selection completed for all three models.\n');

%% Neural Network Parameters
hiddenLayerSize = [20 20];  % simple 2-hidden-layer MLP

%% Train and Evaluate Frequency Model
[net_freq, freq_acc, freq_prec, freq_recall, freq_f1, tr_freq] = ...
    train_and_evaluate_nn( ...
        X_freq_train, Y_freq_train, ...
        X_freq_val,   Y_freq_val, ...
        X_freq_test,  Y_freq_test, ...
        hiddenLayerSize);

fprintf('\n=== Frequency Model Performance ===\n');
fprintf('Accuracy         : %.2f %%\n', freq_acc);
fprintf('Avg Precision    : %.2f\n',  freq_prec);
fprintf('Avg Recall       : %.2f\n',  freq_recall);
fprintf('Avg F1-score     : %.2f\n',  freq_f1);

%% Train and Evaluate Time Model
[net_time, time_acc, time_prec, time_recall, time_f1, tr_time] = ...
    train_and_evaluate_nn( ...
        X_time_train, Y_time_train, ...
        X_time_val,   Y_time_val, ...
        X_time_test,  Y_time_test, ...
        hiddenLayerSize);

fprintf('\n=== Time Model Performance ===\n');
fprintf('Accuracy         : %.2f %%\n', time_acc);
fprintf('Avg Precision    : %.2f\n',  time_prec);
fprintf('Avg Recall       : %.2f\n',  time_recall);
fprintf('Avg F1-score     : %.2f\n',  time_f1);

%% Train and Evaluate Time-Frequency Model
[net_timeFreq, tf_acc, tf_prec, tf_recall, tf_f1, tr_timeFreq] = ...
    train_and_evaluate_nn( ...
        X_timeFreq_train, Y_timeFreq_train, ...
        X_timeFreq_val,   Y_timeFreq_val, ...
        X_timeFreq_test,  Y_timeFreq_test, ...
        hiddenLayerSize);

fprintf('\n=== Time-Frequency Model Performance ===\n');
fprintf('Accuracy         : %.2f %%\n', tf_acc);
fprintf('Avg Precision    : %.2f\n',  tf_prec);
fprintf('Avg Recall       : %.2f\n',  tf_recall);
fprintf('Avg F1-score     : %.2f\n\n',  tf_f1);

%% Optionally save models and feature indices
save('Group137_models_and_features.mat', ...
     'net_freq', 'net_time', 'net_timeFreq', ...
     'freqSelectedFeatures', 'timeSelectedFeatures', 'timeFreqSelectedFeatures', ...
     'tr_freq', 'tr_time', 'tr_timeFreq');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                         LOCAL HELPER FUNCTIONS                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [X_train, Y_train_onehot, X_val, Y_val_onehot, X_test, Y_test_onehot] = ...
         split_dataset(X, Y, train_ratio, val_ratio, numClasses)
% SPLIT_DATASET
%   Splits data X (features x samples) and numeric labels Y (1 x samples)
%   into train / validation / test sets, and converts labels to one-hot.

    N = size(X, 2);
    idx = randperm(N);

    train_end = floor(train_ratio * N);
    val_end   = floor((train_ratio + val_ratio) * N);

    train_idx = idx(1:train_end);
    val_idx   = idx(train_end+1:val_end);
    test_idx  = idx(val_end+1:end);

    X_train = X(:, train_idx);
    Y_train = Y(train_idx);
    Y_train_onehot = full(ind2vec(Y_train, numClasses));

    X_val   = X(:, val_idx);
    Y_val   = Y(val_idx);
    Y_val_onehot = full(ind2vec(Y_val, numClasses));

    X_test  = X(:, test_idx);
    Y_test  = Y(test_idx);
    Y_test_onehot = full(ind2vec(Y_test, numClasses));
end


function [net, accuracy, avg_precision, avg_recall, avg_f1, tr] = ...
         train_and_evaluate_nn(X_train, Y_train, X_val, Y_val, X_test, Y_test, hiddenLayerSize)
% TRAIN_AND_EVALUATE_NN
%   Trains a feedforwardnet classifier and returns performance metrics.

    % Combine all sets for MATLAB's divideind
    X_all = [X_train, X_val, X_test];
    Y_all = [Y_train, Y_val, Y_test];

    trainSize = size(X_train, 2);
    valSize   = size(X_val, 2);
    testSize  = size(X_test, 2);
    totalSize = trainSize + valSize + testSize;

    net = feedforwardnet(hiddenLayerSize, 'trainscg');
    net.trainParam.showWindow = true;

    % Use indices for train/val/test
    net.divideFcn = 'divideind';
    net.divideParam.trainInd = 1:trainSize;
    net.divideParam.valInd   = (trainSize+1):(trainSize+valSize);
    net.divideParam.testInd  = (trainSize+valSize+1):totalSize;

    % Train
    [net, tr] = train(net, X_all, Y_all);

    % Test set evaluation
    Y_pred = net(X_test);
    [~, Y_pred_class] = max(Y_pred);
    [~, Y_true_class] = max(Y_test);

    accuracy = sum(Y_pred_class == Y_true_class) / numel(Y_true_class) * 100;

    % Confusion matrix
    confMat    = confusionmat(Y_true_class, Y_pred_class);
    numClasses = size(Y_test, 1);

    precision = zeros(numClasses,1);
    recall    = zeros(numClasses,1);
    f1        = zeros(numClasses,1);

    for c = 1:numClasses
        tp = confMat(c,c);
        fp = sum(confMat(:,c)) - tp;
        fn = sum(confMat(c,:)) - tp;
        precision(c) = tp / (tp + fp + eps);
        recall(c)    = tp / (tp + fn + eps);
        f1(c)        = 2 * (precision(c) * recall(c)) / (precision(c) + recall(c) + eps);
    end

    avg_precision = mean(precision);
    avg_recall    = mean(recall);
    avg_f1        = mean(f1);

    % Plot confusion matrix (one-hot to categorical)
    figure;
    plotconfusion(Y_test, Y_pred, 'Test Set Confusion Matrix');

    % Plot performance curves (train / val / test loss)
    figure;
    plotperform(tr);
end
