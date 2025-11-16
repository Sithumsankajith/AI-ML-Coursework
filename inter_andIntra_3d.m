%% inter_andintra_3d.m
% 3D variance analysis: intra-user and inter-user feature variability

clear all;
close all;
clc;

rng(100);  % For reproducibility

% Define script params
userRange_min = 1;
userRange_max = 10;
numUsers      = userRange_max - userRange_min + 1;

% 1. Data Loading and Preprocessing
filePatternsTrain = 'Acc_TimeD_FreqD_FDay';
filePatternsTest  = 'Acc_TimeD_FreqD_MDay';

dataDir = 'dataset';   % <--- IMPORTANT: your .mat files folder

fprintf('Loading data for each user...\n');

% Preallocate struct array
userData = struct('trainFeatures', [], 'testFeatures', []);
userData = repmat(userData, 1, userRange_max);

for user = userRange_min:userRange_max
    userStr  = sprintf('U%02d', user);
    trainFile = fullfile(dataDir, [userStr '_' filePatternsTrain '.mat']);
    testFile  = fullfile(dataDir, [userStr '_' filePatternsTest  '.mat']);

    if exist(trainFile, 'file') && exist(testFile, 'file')
        trainData = load(trainFile);
        testData  = load(testFile);

        userData(user).trainFeatures = trainData.(char(fieldnames(trainData)));
        userData(user).testFeatures  = testData.(char(fieldnames(testData)));

        [r, c] = size(userData(user).trainFeatures);
        fprintf('User %d: %d train samples x %d features\n', user, r, c);
    else
        fprintf('Missing data files for user %d\n', user);
        fprintf('  Tried to load:\n');
        fprintf('    Train: %s\n', trainFile);
        fprintf('    Test : %s\n', testFile);
    end
end

%% 2. Data Processing and Variance Analysis

% Check that we have at least one valid user
hasData = ~arrayfun(@(u) isempty(u.trainFeatures), userData);
if ~any(hasData)
    error('No valid trainFeatures found for any user. Check your dataset folder and file names.');
end

% Use the first valid user to determine feature dimension
firstUserIdx   = find(hasData, 1, 'first');
num_features   = size(userData(firstUserIdx).trainFeatures, 2);

% For robustness, use the minimum number of samples across users
num_samples = inf;
for user = userRange_min:userRange_max
    if ~isempty(userData(user).trainFeatures)
        num_samples = min(num_samples, size(userData(user).trainFeatures, 1));
    end
end

if ~isfinite(num_samples) || num_samples == 0
    error('Could not determine a valid number of samples for variance analysis.');
end

fprintf('\nUsing %d samples x %d features per user for variance analysis.\n', ...
        num_samples, num_features);

% Create 3D array [users x samples x features]
all_data = zeros(numUsers, num_samples, num_features);

userIdx = 0;
for user = userRange_min:userRange_max
    userIdx = userIdx + 1;   % index 1..numUsers

    if isempty(userData(user).trainFeatures)
        error('User %d has no trainFeatures. Cannot include in all_data.', user);
    end

    thisData = userData(user).trainFeatures;

    % Ensure we only take the first num_samples rows and correct feature count
    if size(thisData,2) ~= num_features
        error('User %d has different feature dimension (%d) than expected (%d).', ...
               user, size(thisData,2), num_features);
    end

    all_data(userIdx,:,:) = thisData(1:num_samples, :);
end

% Calculate mean across samples for each user -> [users x features]
user_means = squeeze(mean(all_data, 2));  

% Calculate inter-user variance for each feature -> [1 x features]
inter_user_variance = var(user_means, 0, 1);

%% 3. Visualization

output_dir = '3d_analysis';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% 3.1 Intra-user variances: variance of each feature per user over samples
user_variances = zeros(numUsers, num_features);
for u = 1:numUsers
    user_variances(u,:) = var(squeeze(all_data(u,:,:)), 0, 1);
end

% 3.1 3D Surface plot of feature variances across users
figure('Name', 'User Feature Variances (Surface)');
[X, Y] = meshgrid(1:num_features, 1:numUsers);
surf(X, Y, user_variances);
xlabel('Feature Index');
ylabel('User Index');
zlabel('Variance');
title('Feature Variances Across Users (Intra-user)');
colormap(jet);
colorbar;
view(45, 30);
grid on;

saveas(gcf, fullfile(output_dir, 'feature_variances_surface.fig'));
saveas(gcf, fullfile(output_dir, 'feature_variances_surface.png'));

% 3.2 3D Bar plot of feature variances across users
figure('Name', 'User Feature Variances (Bar3)');
bar3(user_variances);
xlabel('Feature Index');
ylabel('User Index');
zlabel('Variance');
title('Feature Variances by User (Intra-user)');
colormap(hot);
colorbar;
view(45, 30);
grid on;

saveas(gcf, fullfile(output_dir, 'feature_variances_bar.fig'));
saveas(gcf, fullfile(output_dir, 'feature_variances_bar.png'));

% 3.3 Mean feature variance (intra-user) + inter-user variance intuition
figure('Name', 'Mean Feature Variances');
plot(mean(user_variances, 1), 'LineWidth', 2);
hold on;
plot(inter_user_variance, '--', 'LineWidth', 2);
xlabel('Feature Index');
ylabel('Variance');
title('Average Intra-user vs Inter-user Feature Variance');
legend('Mean Intra-user Variance', 'Inter-user Variance');
grid on;

saveas(gcf, fullfile(output_dir, 'mean_feature_variances.fig'));
saveas(gcf, fullfile(output_dir, 'mean_feature_variances.png'));

fprintf('\nInter/Intra variance analysis completed and figures saved in "%s".\n', output_dir);
