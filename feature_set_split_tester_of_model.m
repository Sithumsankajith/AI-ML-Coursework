%% feature_set_split_tester_of_model_clean.m
% Compare different feature sets and train/test scenarios for FFNN

clear all; close all; clc;
rng(100);

%% Config
userRange_min = 1;
userRange_max = 10;
numUsers      = userRange_max - userRange_min + 1;

dropoutRate             = 0.3;
l2RegParam              = 1e-4;
performanceGoal         = 1e-5;
minGrad                 = 1e-6;
earlyStoppingPatience   = 10;
maxEpochs               = 500;
learningRate            = 0.01;

featureSets = { ...
  {'Acc_TimeD_FreqD_FDay', 'Acc_TimeD_FreqD_MDay'}, ...
  {'Acc_TimeD_FDay',       'Acc_TimeD_MDay'}, ...
  {'Acc_FreqD_FDay',       'Acc_FreqD_MDay'} ...
  };

featureSetNames = {'TimeD+FreqD', 'TimeD', 'FreqD'};

scenarioNames = {'50-50 Split FDay', 'FDay-MDay Split', 'Combined 50-50 Split'};
dataFolder    = 'dataset';

allResults = cell(length(featureSets), length(scenarioNames));

%% Loop over feature sets and scenarios
for setIdx = 1:length(featureSets)
    fprintf('\n\nProcessing feature set: %s\n', featureSets{setIdx}{1});
    
    for scenarioIdx = 1:length(scenarioNames)
        fprintf('\nScenario: %s\n', scenarioNames{scenarioIdx});
        
        userMetrics = zeros(userRange_max, 14);
        
        for targetUser = userRange_min:userRange_max
            %% Build train/test for this user, feature set & scenario
            userStr = sprintf('U%02d', targetUser);
            
            switch scenarioIdx
                case 1  % 50-50 Split FDay
                    fDayFile = fullfile(dataFolder, [userStr '_' featureSets{setIdx}{1} '.mat']);
                    data = load(fDayFile);
                    allData = data.(char(fieldnames(data)));
                    splitIdx = floor(size(allData, 1)/2);
                    trainFeatures = allData(1:splitIdx,:);
                    testFeatures  = allData(splitIdx+1:end,:);
                    
                case 2  % FDay-MDay Split
                    fDayFile = fullfile(dataFolder, [userStr '_' featureSets{setIdx}{1} '.mat']);
                    mDayFile = fullfile(dataFolder, [userStr '_' featureSets{setIdx}{2} '.mat']);
                    dTrain = load(fDayFile);
                    dTest  = load(mDayFile);
                    trainFeatures = dTrain.(char(fieldnames(dTrain)));
                    testFeatures  = dTest.(char(fieldnames(dTest)));
                    
                case 3  % Combined 50-50 Split of FDay+MDay
                    fDayFile = fullfile(dataFolder, [userStr '_' featureSets{setIdx}{1} '.mat']);
                    mDayFile = fullfile(dataFolder, [userStr '_' featureSets{setIdx}{2} '.mat']);
                    d1 = load(fDayFile);
                    d2 = load(mDayFile);
                    allData = [d1.(char(fieldnames(d1))); d2.(char(fieldnames(d2)))];
                    splitIdx = floor(size(allData, 1)/2);
                    trainFeatures = allData(1:splitIdx,:);
                    testFeatures  = allData(splitIdx+1:end,:);
            end
            
            % Positive training samples (target user)
            targetSamples = trainFeatures;
            targetLabels  = ones(size(targetSamples,1),1);
            numTargetSamples = size(targetSamples,1);
            
            %% Collect negative (imposter) training samples
            impFeatTrain = [];
            impLblTrain  = [];
            samplesPerImposter = ceil(numTargetSamples/(numUsers-1));
            
            for impUser = userRange_min:userRange_max
                if impUser == targetUser, continue; end
                
                impStr = sprintf('U%02d', impUser);
                switch scenarioIdx
                    case 1 % 50-50 FDay
                        impFile = fullfile(dataFolder, [impStr '_' featureSets{setIdx}{1} '.mat']);
                        if ~exist(impFile,'file'), continue; end
                        d = load(impFile);
                        impData = d.(char(fieldnames(d)));
                        impData = impData(1:splitIdx,:);
                        
                    case 2 % FDay-MDay
                        impFile = fullfile(dataFolder, [impStr '_' featureSets{setIdx}{1} '.mat']);
                        if ~exist(impFile,'file'), continue; end
                        d = load(impFile);
                        impData = d.(char(fieldnames(d)));
                        
                    case 3 % Combined 50-50
                        impFile1 = fullfile(dataFolder, [impStr '_' featureSets{setIdx}{1} '.mat']);
                        impFile2 = fullfile(dataFolder, [impStr '_' featureSets{setIdx}{2} '.mat']);
                        if ~exist(impFile1,'file') || ~exist(impFile2,'file'), continue; end
                        d1 = load(impFile1); d2 = load(impFile2);
                        allImp = [d1.(char(fieldnames(d1))); d2.(char(fieldnames(d2)))];
                        impData = allImp(1:splitIdx,:);
                end
                
                nTake = min(samplesPerImposter, size(impData,1));
                impFeatTrain = [impFeatTrain; impData(1:nTake,:)];
                impLblTrain  = [impLblTrain; zeros(nTake,1)];
            end
            
            X_train = [targetSamples; impFeatTrain];
            y_train = [targetLabels; impLblTrain];
            
            idx = randperm(size(X_train,1));
            X_train = X_train(idx,:);
            y_train = y_train(idx,:);
            
            X_train = normalize(X_train, 'range');
            
            %% Build test set: positives + negatives
            testLabelsPos = ones(size(testFeatures,1),1);
            
            impFeatTest = [];
            impLblTest  = [];
            samplesPerImposter = ceil(size(testFeatures,1)/(numUsers-1));
            
            for impUser = userRange_min:userRange_max
                if impUser == targetUser, continue; end
                impStr = sprintf('U%02d', impUser);
                
                switch scenarioIdx
                    case 1
                        impFile = fullfile(dataFolder, [impStr '_' featureSets{setIdx}{1} '.mat']);
                        if ~exist(impFile,'file'), continue; end
                        d = load(impFile);
                        impData = d.(char(fieldnames(d)));
                        impData = impData(splitIdx+1:end,:);
                        
                    case 2
                        impFile = fullfile(dataFolder, [impStr '_' featureSets{setIdx}{2} '.mat']);
                        if ~exist(impFile,'file'), continue; end
                        d = load(impFile);
                        impData = d.(char(fieldnames(d)));
                        
                    case 3
                        impFile1 = fullfile(dataFolder, [impStr '_' featureSets{setIdx}{1} '.mat']);
                        impFile2 = fullfile(dataFolder, [impStr '_' featureSets{setIdx}{2} '.mat']);
                        if ~exist(impFile1,'file') || ~exist(impFile2,'file'), continue; end
                        d1 = load(impFile1); d2 = load(impFile2);
                        allImp = [d1.(char(fieldnames(d1))); d2.(char(fieldnames(d2)))];
                        impData = allImp(splitIdx+1:end,:);
                end
                
                nTake = min(samplesPerImposter, size(impData,1));
                idxImp = randperm(size(impData,1), nTake);
                impFeatTest = [impFeatTest; impData(idxImp,:)];
                impLblTest  = [impLblTest;  zeros(nTake,1)];
            end
            
            X_test = [testFeatures; impFeatTest];
            y_test = [testLabelsPos; impLblTest];
            
            X_test = normalize(X_test, 'range');
            
            %% Neural network
            net = feedforwardnet(131, 'trainscg');
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
            
            % Train
            net = train(net, X_train', y_train');
            
            % Scores + predictions
            scores = net(X_test')';
            yPred  = double(scores > 0.5);
            
            % Metrics
            yTrueBin = y_test == 1;
            predBin  = yPred == 1;
            
            tp = sum(predBin &  yTrueBin);
            fp = sum(predBin & ~yTrueBin);
            fn = sum(~predBin &  yTrueBin);
            tn = sum(~predBin & ~yTrueBin);
            
            accuracy   = sum(predBin == yTrueBin)/numel(yTrueBin);
            precision  = tp/(tp+fp+eps);
            recall     = tp/(tp+fn+eps);
            specificity= tn/(tn+fp+eps);
            f1_score   = 2*precision*recall/(precision+recall+eps);
            mcc_num    = (tp*tn - fp*fn);
            mcc_den    = sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));
            mcc        = mcc_num/(mcc_den+eps);
            
            % ROC / FAR / FRR / EER using scores
            [FAR, TPR, ~, auc] = perfcurve(yTrueBin, scores, true);
            FRR = 1 - TPR;
            [~, eerIdx] = min(abs(FAR - FRR));
            EER = (FAR(eerIdx) + FRR(eerIdx))/2;
            
            userMetrics(targetUser,:) = [ ...
                accuracy, precision, recall, specificity, ...
                f1_score, mcc, FAR(eerIdx)*100, FRR(eerIdx)*100, EER*100, auc, ...
                size(X_train,1), size(targetSamples,1), size(impFeatTrain,1), ...
                size(X_test,1)];
        end
        
        avgMetrics = mean(userMetrics,1);
        allResults{setIdx, scenarioIdx} = struct( ...
            'Scenario',   scenarioNames{scenarioIdx}, ...
            'FeatureSet', featureSets{setIdx}{1}, ...
            'Accuracy',   avgMetrics(1)*100, ...
            'Precision',  avgMetrics(2)*100, ...
            'Recall',     avgMetrics(3)*100, ...
            'Specificity',avgMetrics(4)*100, ...
            'F1Score',    avgMetrics(5)*100, ...
            'MCC',        avgMetrics(6), ...
            'FAR',        avgMetrics(7), ...
            'FRR',        avgMetrics(8), ...
            'EER',        avgMetrics(9), ...
            'AUC',        avgMetrics(10));
    end
end

%% Bar charts for key metrics
metrics = {'FAR','FRR','EER','Accuracy','F1Score'};
titles  = {'False Acceptance Rate','False Rejection Rate','Equal Error Rate','Accuracy','F1-score'};

figure('Position',[100 100 1200 800]);

for i = 1:length(metrics)
    subplot(2,3,i);
    data = zeros(length(featureSets), length(scenarioNames));
    for setIdx = 1:length(featureSets)
        for scenIdx = 1:length(scenarioNames)
            data(setIdx,scenIdx) = allResults{setIdx,scenIdx}.(metrics{i});
        end
    end
    bar(data);
    title(titles{i});
    xlabel('Feature Sets');
    ylabel([titles{i} ' (%)']);
    xticklabels(featureSetNames);
    legend(scenarioNames,'Location','best');
    grid on;
end

sgtitle('Performance Across Scenarios and Feature Sets');

%% Console summary
metricsList = {'Accuracy','Precision','Recall','Specificity','F1Score','MCC','FAR','FRR','EER','AUC'};

fprintf('\n=== Performance Metrics Comparison ===\n\n');
for setIdx = 1:length(featureSetNames)
    fprintf('\n=== %s Domain ===\n', featureSetNames{setIdx});
    fprintf('%-12s', 'Metric');
    for scenIdx = 1:length(scenarioNames)
        fprintf('%-20s', scenarioNames{scenIdx});
    end
    fprintf('\n%s\n', repmat('-', 1, 12 + 20*length(scenarioNames)));
    
    for m = 1:length(metricsList)
        metricName = metricsList{m};
        fprintf('%-12s', metricName);
        for scenIdx = 1:length(scenarioNames)
            val = allResults{setIdx,scenIdx}.(metricName);
            if strcmp(metricName,'MCC') || strcmp(metricName,'AUC')
                fprintf('%-20.4f', val);
            else
                fprintf('%6.2f%%             ', val);
            end
        end
        fprintf('\n');
    end
    fprintf('\n');
end

save('testing_scenarios_results_clean.mat', 'allResults');
