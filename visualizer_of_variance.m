%% Feature Variance Visualizer (Robust Version)
% Analyzes intra-user (within-user) and inter-user (between-user)
% variance to find stable and discriminative features.

clear all; close all; clc;

folder_path = 'dataset'; % Folder with feature MATs
num_users   = 10;

fprintf('Starting variance analysis...\n');

%% ---------- 1. TIME DOMAIN (TD) ANALYSIS ----------

fprintf('Processing Time Domain (TD) features...\n');
mean_vals_TD_per_user      = [];
std_vals_TD_per_user       = [];
intra_variance_TD_per_user = [];
all_means_TD               = [];

for user_idx = 1:num_users
    file1 = sprintf('U%02d_Acc_TimeD_FDay.mat', user_idx);
    file2 = sprintf('U%02d_Acc_TimeD_MDay.mat', user_idx);
    file1_path = fullfile(folder_path, file1);
    file2_path = fullfile(folder_path, file2);

    if ~exist(file1_path,'file') || ~exist(file2_path,'file')
        fprintf('WARNING: Missing TD files for user %d. Skipping user.\n', user_idx);
        continue;
    end

    data1 = load(file1_path);
    data2 = load(file2_path);

    feature_data1 = data1.(char(fieldnames(data1)));
    feature_data2 = data2.(char(fieldnames(data2)));

    user_data_TD = [feature_data1; feature_data2];

    if isempty(user_data_TD)
        fprintf('WARNING: No TD data for user %d after concatenation.\n', user_idx);
        continue;
    end

    user_mean = mean(user_data_TD, 1);
    mean_vals_TD_per_user(user_idx, :)      = user_mean;
    std_vals_TD_per_user(user_idx, :)       = std(user_data_TD, 0, 1);
    intra_variance_TD_per_user(user_idx, :) = var(user_data_TD, 0, 1);

    all_means_TD(user_idx, :) = user_mean;
end

% Remove any all-zero rows if some users were skipped
validRows = any(mean_vals_TD_per_user, 2);
mean_vals_TD_per_user      = mean_vals_TD_per_user(validRows, :);
std_vals_TD_per_user       = std_vals_TD_per_user(validRows, :);
intra_variance_TD_per_user = intra_variance_TD_per_user(validRows, :);
all_means_TD               = all_means_TD(validRows, :);

inter_variance_TD      = var(all_means_TD, 0, 1);
intra_variance_TD_mean = mean(intra_variance_TD_per_user, 1);
std_vals_TD            = mean(std_vals_TD_per_user, 1);

%% ---------- 2. FREQUENCY DOMAIN (FD) ANALYSIS ----------

fprintf('Processing Frequency Domain (FD) features...\n');
mean_vals_FD_per_user      = [];
std_vals_FD_per_user       = [];
intra_variance_FD_per_user = [];
all_means_FD               = [];

for user_idx = 1:num_users
    file3 = sprintf('U%02d_Acc_FreqD_FDay.mat', user_idx);
    file4 = sprintf('U%02d_Acc_FreqD_MDay.mat', user_idx);
    file3_path = fullfile(folder_path, file3);
    file4_path = fullfile(folder_path, file4);

    if ~exist(file3_path,'file') || ~exist(file4_path,'file')
        fprintf('WARNING: Missing FD files for user %d. Skipping user.\n', user_idx);
        continue;
    end

    data3 = load(file3_path);
    data4 = load(file4_path);

    feature_data3 = data3.(char(fieldnames(data3)));
    feature_data4 = data4.(char(fieldnames(data4)));

    user_data_FD = [feature_data3; feature_data4];

    if isempty(user_data_FD)
        fprintf('WARNING: No FD data for user %d after concatenation.\n', user_idx);
        continue;
    end

    user_mean = mean(user_data_FD, 1);
    mean_vals_FD_per_user(user_idx, :)      = user_mean;
    std_vals_FD_per_user(user_idx, :)       = std(user_data_FD, 0, 1);
    intra_variance_FD_per_user(user_idx, :) = var(user_data_FD, 0, 1);

    all_means_FD(user_idx, :) = user_mean;
end

validRows = any(mean_vals_FD_per_user, 2);
mean_vals_FD_per_user      = mean_vals_FD_per_user(validRows, :);
std_vals_FD_per_user       = std_vals_FD_per_user(validRows, :);
intra_variance_FD_per_user = intra_variance_FD_per_user(validRows, :);
all_means_FD               = all_means_FD(validRows, :);

inter_variance_FD      = var(all_means_FD, 0, 1);
intra_variance_FD_mean = mean(intra_variance_FD_per_user, 1);
std_vals_FD            = mean(std_vals_FD_per_user, 1);

%% ---------- 3. TIME-FREQUENCY DOMAIN (TDFD) ANALYSIS ----------

fprintf('Processing Time-Frequency Domain (TDFD) features...\n');
mean_vals_TDFD_per_user      = [];
std_vals_TDFD_per_user       = [];
intra_variance_TDFD_per_user = [];
all_means_TDFD               = [];

for user_idx = 1:num_users
    file5 = sprintf('U%02d_Acc_TimeD_FreqD_FDay.mat', user_idx);
    file6 = sprintf('U%02d_Acc_TimeD_FreqD_MDay.mat', user_idx);
    file5_path = fullfile(folder_path, file5);
    file6_path = fullfile(folder_path, file6);

    if ~exist(file5_path,'file') || ~exist(file6_path,'file')
        fprintf('WARNING: Missing TDFD files for user %d. Skipping user.\n', user_idx);
        continue;
    end

    data5 = load(file5_path);
    data6 = load(file6_path);

    feature_data5 = data5.(char(fieldnames(data5)));
    feature_data6 = data6.(char(fieldnames(data6)));

    user_data_TDFD = [feature_data5; feature_data6];

    if isempty(user_data_TDFD)
        fprintf('WARNING: No TDFD data for user %d after concatenation.\n', user_idx);
        continue;
    end

    user_mean = mean(user_data_TDFD, 1);
    mean_vals_TDFD_per_user(user_idx, :)      = user_mean;
    std_vals_TDFD_per_user(user_idx, :)       = std(user_data_TDFD, 0, 1);
    intra_variance_TDFD_per_user(user_idx, :) = var(user_data_TDFD, 0, 1);

    all_means_TDFD(user_idx, :) = user_mean;
end

validRows = any(mean_vals_TDFD_per_user, 2);
mean_vals_TDFD_per_user      = mean_vals_TDFD_per_user(validRows, :);
std_vals_TDFD_per_user       = std_vals_TDFD_per_user(validRows, :);
intra_variance_TDFD_per_user = intra_variance_TDFD_per_user(validRows, :);
all_means_TDFD               = all_means_TDFD(validRows, :);

inter_variance_TDFD      = var(all_means_TDFD, 0, 1);
intra_variance_TDFD_mean = mean(intra_variance_TDFD_per_user, 1);
std_vals_TDFD            = mean(std_vals_TDFD_per_user, 1);

%% ---------- 4. PLOTTING ----------

fprintf('Generating plots...\n');

% Helper for legend labels
userLabels = arrayfun(@(u) sprintf('U%d', u), 1:num_users, 'UniformOutput', false);
userLabels{end+1} = 'Mean';

% Time Domain
figure('Name', 'Time Domain Analysis', 'Position', [100 100 900 700]);

subplot(3,1,1);
hold on;
plot(intra_variance_TD_per_user', 'LineWidth', 1);
plot(intra_variance_TD_mean, 'k--', 'LineWidth', 2);
hold off;
title('TD Intra-variance (Low is good)');
xlabel('Feature Index'); ylabel('Variance');
legend(userLabels, 'Location','bestoutside');
grid on;
xlim([1 length(inter_variance_TD)]);

subplot(3,1,2);
hold on;
plot(std_vals_TD_per_user', 'LineWidth', 1);
plot(std_vals_TD, 'k--', 'LineWidth', 2);
hold off;
title('TD Standard Deviation');
xlabel('Feature Index'); ylabel('Std. Dev.');
legend(userLabels, 'Location','bestoutside');
grid on;
xlim([1 length(inter_variance_TD)]);

subplot(3,1,3);
plot(inter_variance_TD, 'r-', 'LineWidth', 1.5);
title('TD Inter-variance (High is good)');
xlabel('Feature Index'); ylabel('Variance');
grid on;
xlim([1 length(inter_variance_TD)]);

% Frequency Domain
figure('Name', 'Frequency Domain Analysis', 'Position', [150 150 900 700]);

subplot(3,1,1);
hold on;
plot(intra_variance_FD_per_user', 'LineWidth', 1);
plot(intra_variance_FD_mean, 'k--', 'LineWidth', 2);
hold off;
title('FD Intra-variance (Low is good)');
xlabel('Feature Index'); ylabel('Variance');
legend(userLabels, 'Location','bestoutside');
grid on;
xlim([1 length(inter_variance_FD)]);

subplot(3,1,2);
hold on;
plot(std_vals_FD_per_user', 'LineWidth', 1);
plot(std_vals_FD, 'k--', 'LineWidth', 2);
hold off;
title('FD Standard Deviation');
xlabel('Feature Index'); ylabel('Std. Dev.');
legend(userLabels, 'Location','bestoutside');
grid on;
xlim([1 length(inter_variance_FD)]);

subplot(3,1,3);
plot(inter_variance_FD, 'r-', 'LineWidth', 1.5);
title('FD Inter-variance (High is good)');
xlabel('Feature Index'); ylabel('Variance');
grid on;
xlim([1 length(inter_variance_FD)]);

% Time-Frequency Domain
figure('Name', 'Time-Frequency Domain Analysis', 'Position', [200 200 900 700]);

subplot(3,1,1);
hold on;
plot(intra_variance_TDFD_per_user', 'LineWidth', 1);
plot(intra_variance_TDFD_mean, 'k--', 'LineWidth', 2);
hold off;
title('TDFD Intra-variance (Low is good)');
xlabel('Feature Index'); ylabel('Variance');
legend(userLabels, 'Location','bestoutside');
grid on;
xlim([1 length(inter_variance_TDFD)]);

subplot(3,1,2);
hold on;
plot(std_vals_TDFD_per_user', 'LineWidth', 1);
plot(std_vals_TDFD, 'k--', 'LineWidth', 2);
hold off;
title('TDFD Standard Deviation');
xlabel('Feature Index'); ylabel('Std. Dev.');
legend(userLabels, 'Location','bestoutside');
grid on;
xlim([1 length(inter_variance_TDFD)]);

subplot(3,1,3);
plot(inter_variance_TDFD, 'r-', 'LineWidth', 1.5);
title('TDFD Inter-variance (High is good)');
xlabel('Feature Index'); ylabel('Variance');
grid on;
xlim([1 length(inter_variance_TDFD)]);

fprintf('Analysis complete. All plots generated.\n');
