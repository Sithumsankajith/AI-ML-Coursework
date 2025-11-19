%% User Similarity via DTW on Acc_TimeD_FreqD Features
% Computes a DTW-based similarity matrix between users using the
% combined Time + Frequency domain accelerometer features.

clear all; close all; clc;

folder_path = 'dataset';
num_users   = 10;

fprintf('Computing DTW-based user similarity on Acc_TimeD_FreqD features...\n');

%% 1. Load data for all users (FDay + MDay) and stack for normalization

user_data_raw = cell(num_users, 1);  % each cell: [Ni x D] for user i
all_data      = [];                  % stacked data from all users

for user_idx = 1:num_users
    file1 = sprintf('U%02d_Acc_TimeD_FreqD_FDay.mat', user_idx);
    file2 = sprintf('U%02d_Acc_TimeD_FreqD_MDay.mat', user_idx);

    file1_path = fullfile(folder_path, file1);
    file2_path = fullfile(folder_path, file2);

    if ~exist(file1_path, 'file') || ~exist(file2_path, 'file')
        fprintf('WARNING: Missing files for user %d. Skipping this user.\n', user_idx);
        user_data_raw{user_idx} = [];
        continue;
    end

    data1 = load(file1_path);
    data2 = load(file2_path);

    feat1 = data1.(char(fieldnames(data1)));
    feat2 = data2.(char(fieldnames(data2)));

    user_data = [feat1; feat2];  % concatenate days

    if isempty(user_data)
        fprintf('WARNING: No data for user %d after concatenation.\n', user_idx);
        user_data_raw{user_idx} = [];
        continue;
    end

    user_data_raw{user_idx} = user_data;
    all_data = [all_data; user_data];  % accumulate for global normalization
end

% Safety check
if isempty(all_data)
    error('No data loaded from any user. Check the dataset folder and file names.');
end

%% 2. Global z-score normalization (across all users)

mean_vals = mean(all_data, 1);
std_vals  = std(all_data, 0, 1);
std_vals(std_vals == 0) = 1;   % avoid division by zero

user_data_norm = cell(num_users,1);
for user_idx = 1:num_users
    X = user_data_raw{user_idx};
    if isempty(X)
        user_data_norm{user_idx} = [];
        continue;
    end
    user_data_norm{user_idx} = (X - mean_vals) ./ std_vals;
end

%% 3. DTW distance matrix between users

dtw_matrix = inf(num_users, num_users);  % initialize with Inf for safety

for user_idx_1 = 1:num_users
    X1 = user_data_norm{user_idx_1};
    if isempty(X1)
        continue;
    end

    seq1 = X1(:);  % flatten to 1D sequence

    for user_idx_2 = user_idx_1:num_users
        X2 = user_data_norm{user_idx_2};
        if isempty(X2)
            continue;
        end

        seq2 = X2(:);

        % Compute DTW distance between flattened sequences
        [dist, ~, ~] = dtw(seq1, seq2);

        dtw_matrix(user_idx_1, user_idx_2) = dist;
        dtw_matrix(user_idx_2, user_idx_1) = dist;  % symmetric

        fprintf('DTW distance between User %d and User %d: %.4f\n', ...
            user_idx_1, user_idx_2, dist);
    end
end

%% 4. Most similar user (lowest DTW distance) per user

most_similar_users = cell(num_users, 1);

for user_idx = 1:num_users
    row = dtw_matrix(user_idx, :);

    % skip users with no data (row all Inf)
    if all(isinf(row))
        most_similar_users{user_idx} = struct( ...
            'User', user_idx, ...
            'MostSimilarUser', NaN, ...
            'Distance', Inf);
        continue;
    end

    % ignore self-distance
    row(user_idx) = Inf;

    [min_dist, similar_user_idx] = min(row);

    most_similar_users{user_idx} = struct( ...
        'User', user_idx, ...
        'MostSimilarUser', similar_user_idx, ...
        'Distance', min_dist);
end

%% 5. Heatmap visualization with annotations

figure('Position', [100 100 800 600]);
imagesc(dtw_matrix);
colorbar;
title('DTW Distance Matrix (Acc\_TimeD\_FreqD Features)');
xlabel('User Index');
ylabel('User Index');
set(gca, 'XTick', 1:num_users, 'YTick', 1:num_users);

[Xg, Yg] = meshgrid(1:num_users, 1:num_users);
hold on;
for targetUser = 1:num_users
    for secondaryUser = 1:num_users
        val = dtw_matrix(targetUser, secondaryUser);
        if isinf(val)
            txt = 'Inf';
        else
            txt = sprintf('%.2f', val);
        end

        text(secondaryUser, targetUser, txt, ...
            'HorizontalAlignment', 'center', ...
            'Color', 'black', ...
            'FontSize', 8);

        if ~isinf(val) && ...
           ~isempty(most_similar_users{targetUser}.MostSimilarUser) && ...
           most_similar_users{targetUser}.MostSimilarUser == secondaryUser
            plot(secondaryUser, targetUser, 'rs', 'MarkerSize', 20, 'LineWidth', 1.5);
        end
    end
end
hold off;

%% 6. Print summary

fprintf('\nMost Similar Users (based on DTW distance):\n');
for user_idx = 1:num_users
    info = most_similar_users{user_idx};
    if isinf(info.Distance) || isnan(info.MostSimilarUser)
        fprintf('User %d: no valid comparison (missing data)\n', info.User);
    else
        fprintf('User %d is most similar to User %d with DTW distance: %.4f\n', ...
            info.User, info.MostSimilarUser, info.Distance);
    end
end
