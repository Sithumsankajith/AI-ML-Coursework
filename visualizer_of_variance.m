%% Feature Variance Visualizer
% This script analyzes intra-user (within-user) and inter-user (between-user)
% variance to find stable and discriminative features.

clear all; close all; clc;

folder_path = 'dataset'; % Use the folder we created
num_users = 10;

fprintf('Starting variance analysis...\n');

% --- 1. TIME DOMAIN (TD) ANALYSIS ---

fprintf('Processing Time Domain (TD) features...\n');
mean_vals_TD_per_user = []; 
std_vals_TD_per_user = [];  
intra_variance_TD_per_user = [];
all_means_TD = []; % Used to calculate inter-variance

for user_idx = 1:num_users
    % Load data for the current user
    file1 = sprintf('U%02d_Acc_TimeD_FDay.mat', user_idx);
    file2 = sprintf('U%02d_Acc_TimeD_MDay.mat', user_idx);
    file1_path = fullfile(folder_path, file1);
    file2_path = fullfile(folder_path, file2);
    
    data1 = load(file1_path);
    data2 = load(file2_path);

    % Extract feature matrices
    feature_data1 = data1.(char(fieldnames(data1)));
    feature_data2 = data2.(char(fieldnames(data2)));

    % Combine all data for this user
    user_data_TD = [feature_data1; feature_data2];
    
    % Calculate and store stats for THIS user
    user_mean = mean(user_data_TD);
    mean_vals_TD_per_user(user_idx, :) = user_mean;
    std_vals_TD_per_user(user_idx, :) = std(user_data_TD);
    intra_variance_TD_per_user(user_idx, :) = var(user_data_TD);
    
    % Store the mean for inter-variance calculation
    all_means_TD(user_idx, :) = user_mean;
end

% Inter-Variance (variance of the means)
inter_variance_TD = var(all_means_TD);
% Mean Intra-Variance (average of all users' variances)
intra_variance_TD_mean = mean(intra_variance_TD_per_user, 1);
% Mean Standard Deviation
std_vals_TD = mean(std_vals_TD_per_user, 1);


% --- 2. FREQUENCY DOMAIN (FD) ANALYSIS ---

fprintf('Processing Frequency Domain (FD) features...\n');
mean_vals_FD_per_user = []; 
std_vals_FD_per_user = [];  
intra_variance_FD_per_user = [];
all_means_FD = []; % Used to calculate inter-variance

for user_idx = 1:num_users
    % Load data for the current user
    file3 = sprintf('U%02d_Acc_FreqD_FDay.mat', user_idx);
    file4 = sprintf('U%02d_Acc_FreqD_MDay.mat', user_idx);
    file3_path = fullfile(folder_path, file3);
    file4_path = fullfile(folder_path, file4);
    
    data3 = load(file3_path);
    data4 = load(file4_path);

    % Extract feature matrices
    feature_data3 = data3.(char(fieldnames(data3)));
    feature_data4 = data4.(char(fieldnames(data4)));

    % Combine all data for this user
    user_data_FD = [feature_data3; feature_data4];
    
    % Calculate and store stats for THIS user
    user_mean = mean(user_data_FD);
    mean_vals_FD_per_user(user_idx, :) = user_mean;
    std_vals_FD_per_user(user_idx, :) = std(user_data_FD);
    intra_variance_FD_per_user(user_idx, :) = var(user_data_FD);
    
    % Store the mean for inter-variance calculation
    all_means_FD(user_idx, :) = user_mean;
end

% Inter-Variance (variance of the means)
inter_variance_FD = var(all_means_FD);
% Mean Intra-Variance (average of all users' variances)
intra_variance_FD_mean = mean(intra_variance_FD_per_user, 1);
% Mean Standard Deviation
std_vals_FD = mean(std_vals_FD_per_user, 1);


% --- 3. TIME-FREQUENCY DOMAIN (TDFD) ANALYSIS ---

fprintf('Processing Time-Frequency Domain (TDFD) features...\n');
mean_vals_TDFD_per_user = []; 
std_vals_TDFD_per_user = [];  
intra_variance_TDFD_per_user = [];
all_means_TDFD = []; % Used to calculate inter-variance

for user_idx = 1:num_users
    % Load data for the current user
    file5 = sprintf('U%02d_Acc_TimeD_FreqD_FDay.mat', user_idx);
    file6 = sprintf('U%02d_Acc_TimeD_FreqD_MDay.mat', user_idx);
    file5_path = fullfile(folder_path, file5);
    file6_path = fullfile(folder_path, file6);
    
    data5 = load(file5_path);
    data6 = load(file6_path);

    % Extract feature matrices
    feature_data5 = data5.(char(fieldnames(data5)));
    feature_data6 = data6.(char(fieldnames(data6)));

    % Combine all data for this user
    user_data_TDFD = [feature_data5; feature_data6];
    
    % Calculate and store stats for THIS user
    user_mean = mean(user_data_TDFD);
    mean_vals_TDFD_per_user(user_idx, :) = user_mean;
    std_vals_TDFD_per_user(user_idx, :) = std(user_data_TDFD);
    intra_variance_TDFD_per_user(user_idx, :) = var(user_data_TDFD);
    
    % Store the mean for inter-variance calculation
    all_means_TDFD(user_idx, :) = user_mean;
end

% Inter-Variance (variance of the means)
inter_variance_TDFD = var(all_means_TDFD);
% Mean Intra-Variance (average of all users' variances)
intra_variance_TDFD_mean = mean(intra_variance_TDFD_per_user, 1);
% Mean Standard Deviation
std_vals_TDFD = mean(std_vals_TDFD_per_user, 1);


% --- 4. PLOTTING ---

fprintf('Generating plots...\n');

% Create figure for Time Domain
figure('Name', 'Time Domain Analysis', 'Position', [100 100 900 700]);
% Subplot 1: Combined intra-variances and mean
subplot(3,1,1);
hold on;
plot(intra_variance_TD_per_user', 'LineWidth', 1);
plot(intra_variance_TD_mean, 'k--', 'LineWidth', 2);
hold off;
title('Intra-variance (Low is good)');
xlabel('Feature Index');
ylabel('Variance');
legend({'U1', 'U2', 'U3', 'U4', 'U5', 'U6', 'U7', 'U8', 'U9', 'U10', 'Mean'});
grid on;
xlim([1 length(inter_variance_TD)]);

% Subplot 2: Standard deviation with mean
subplot(3,1,2);
hold on;
plot(std_vals_TD_per_user', 'LineWidth', 1);
plot(std_vals_TD, 'k--', 'LineWidth', 2);
hold off;
title('Standard Deviation');
xlabel('Feature Index');
ylabel('Std. Dev.');
legend({'U1', 'U2', 'U3', 'U4', 'U5', 'U6', 'U7', 'U8', 'U9', 'U10', 'Mean'});
grid on;
xlim([1 length(inter_variance_TD)]);

% Subplot 3: Inter-variance only
subplot(3,1,3);
plot(inter_variance_TD, 'r-', 'LineWidth', 1.5);
title('Inter-variance (High is good)');
xlabel('Feature Index');
ylabel('Variance');
grid on;
xlim([1 length(inter_variance_TD)]);


% Create figure for Frequency Domain
figure('Name', 'Frequency Domain Analysis', 'Position', [150 150 900 700]);
% Subplot 1: Combined intra-variances and mean
subplot(3,1,1);
hold on;
plot(intra_variance_FD_per_user', 'LineWidth', 1);
plot(intra_variance_FD_mean, 'k--', 'LineWidth', 2);
hold off;
title('Intra-variance (Low is good)');
xlabel('Feature Index');
ylabel('Variance');
legend({'U1', 'U2', 'U3', 'U4', 'U5', 'U6', 'U7', 'U8', 'U9', 'U10', 'Mean'});
grid on;
xlim([1 length(inter_variance_FD)]);

% Subplot 2: Standard deviation with mean
subplot(3,1,2);
hold on;
plot(std_vals_FD_per_user', 'LineWidth', 1);
plot(std_vals_FD, 'k--', 'LineWidth', 2);
hold off;
title('Standard Deviation');
xlabel('Feature Index');
ylabel('Std. Dev.');
legend({'U1', 'U2', 'U3', 'U4', 'U5', 'U6', 'U7', 'U8', 'U9', 'U10', 'Mean'});
grid on;
xlim([1 length(inter_variance_FD)]);

% Subplot 3: Inter-variance only
subplot(3,1,3);
plot(inter_variance_FD, 'r-', 'LineWidth', 1.5);
title('Inter-variance (High is good)');
xlabel('Feature Index');
ylabel('Variance');
grid on;
xlim([1 length(inter_variance_FD)]);


% Create figure for Time-Frequency Domain
figure('Name', 'Time-Frequency Domain Analysis', 'Position', [200 200 900 700]);
% Subplot 1: Combined intra-variances and mean
subplot(3,1,1);
hold on;
plot(intra_variance_TDFD_per_user', 'LineWidth', 1);
plot(intra_variance_TDFD_mean, 'k--', 'LineWidth', 2);
hold off;
title('Intra-variance (Low is good)');
xlabel('Feature Index');
ylabel('Variance');
legend({'U1', 'U2', 'U3', 'U4', 'U5', 'U6', 'U7', 'U8', 'U9', 'U10', 'Mean'});
grid on;
xlim([1 length(inter_variance_TDFD)]);

% Subplot 2: Standard deviation with mean
subplot(3,1,2);
hold on;
plot(std_vals_TDFD_per_user', 'LineWidth', 1);
plot(std_vals_TDFD, 'k--', 'LineWidth', 2);
hold off;
title('Standard Deviation');
xlabel('Feature Index');
ylabel('Std. Dev.');
legend({'U1', 'U2', 'U3', 'U4', 'U5', 'U6', 'U7', 'U8', 'U9', 'U10', 'Mean'});
grid on;
xlim([1 length(inter_variance_TDFD)]);

% Subplot 3: Inter-variance only
subplot(3,1,3);
plot(inter_variance_TDFD, 'r-', 'LineWidth', 1.5);
title('Inter-variance (High is good)');
xlabel('Feature Index');
ylabel('Variance');
grid on;
xlim([1 length(inter_variance_TDFD)]);

fprintf('Analysis complete. All plots generated.\n');


