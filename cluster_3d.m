%% cluster_3d_clean.m
% PCA clustering of user features (3D / 2D / 1D)

clear all; close all; clc;
rng(100);

userRange_min = 1;
userRange_max = 10;
numUsers      = userRange_max - userRange_min + 1;

filePatternsTrain = 'Acc_TimeD_FreqD_FDay';
filePatternsTest  = 'Acc_TimeD_FreqD_MDay';
dataFolder        = 'dataset';

fprintf('Loading data for each user...\n');

userData = struct('trainFeatures', [], 'testFeatures', []);
userData = repmat(userData, 1, userRange_max);

for u = userRange_min:userRange_max
    uStr = sprintf('U%02d', u);
    trainFile = fullfile(dataFolder, [uStr '_' filePatternsTrain '.mat']);
    testFile  = fullfile(dataFolder, [uStr '_' filePatternsTest  '.mat']);
    
    if exist(trainFile,'file') && exist(testFile,'file')
        dTrain = load(trainFile);
        dTest  = load(testFile);
        userData(u).trainFeatures = dTrain.(char(fieldnames(dTrain)));
        userData(u).testFeatures  = dTest.(char(fieldnames(dTest)));
        fprintf('User %d: %d train samples\n', u, size(userData(u).trainFeatures,1));
    else
        fprintf('Missing data for user %d\n', u);
    end
end

%% Stack all training features
allFeatures = [];
userLabels  = [];

for u = userRange_min:userRange_max
    feat = userData(u).trainFeatures;
    if isempty(feat), continue; end
    allFeatures = [allFeatures; feat];
    userLabels  = [userLabels; u*ones(size(feat,1),1)];
end

if isempty(allFeatures)
    error('No features found. Check data files.');
end

fprintf('Total features: %d samples x %d dims\n', size(allFeatures,1), size(allFeatures,2));

%% PCA
[coeff, score, latent] = pca(allFeatures);
numComponents = size(score,2);

figure('Position',[100 100 800 600]);

if numComponents >= 3
    scatter3(score(:,1), score(:,2), score(:,3), 40, userLabels, 'filled');
    zlabel('PC3');
elseif numComponents == 2
    scatter(score(:,1), score(:,2), 40, userLabels, 'filled');
elseif numComponents == 1
    scatter(score(:,1), zeros(size(score,1),1), 40, userLabels, 'filled');
else
    error('PCA did not return any components.');
end

colormap(jet(numUsers));
colorbar;
xlabel('PC1'); ylabel('PC2');
title(sprintf('User Feature Clusters in PCA Space (%d components)', numComponents));
grid on; rotate3d on;

explained = latent/sum(latent)*100;
fprintf('Total explained variance (all PCs): %.2f%%\n', sum(explained));
saveas(gcf,'3d_cluster_clean.fig');
saveas(gcf,'3d_cluster_clean.png');
