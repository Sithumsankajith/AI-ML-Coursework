%% cosine_similarity.m
% Point-wise cosine similarity between FDay and MDay samples (subsampled)
clear all;
close all;
clc;

rng(100);  % For reproducibility

%% ================== CONFIGURATION ======================================
userRange_min   = 1;
userRange_max   = 10;

filePatternsTrain = 'Acc_TimeD_FreqD_FDay';
filePatternsTest  = 'Acc_TimeD_FreqD_MDay';

dataDir          = 'dataset';   % folder where your Uxx_*.mat are stored

MAX_SAMPLES_PER_USER = 400;     % <-- reduce if still heavy, increase if PC is strong
PLOT_SIM_THRESHOLD   = 0.5;     % only plot points above this similarity in 3D

%% ================== LOAD DATA ==========================================
fprintf('Loading data for each user...\n');

% Pre-create a struct with fields user1, user2, ...
userData = struct();

for user = userRange_min:userRange_max
    userStr   = sprintf('U%02d', user);

    trainFile = fullfile(dataDir, [userStr '_' filePatternsTrain '.mat']);
    testFile  = fullfile(dataDir, [userStr '_' filePatternsTest  '.mat']);

    if exist(trainFile, 'file') && exist(testFile, 'file')
        trainData = load(trainFile);
        testData  = load(testFile);

        % assume each .mat contains a single variable: [N x D]
        fTrain = trainData.(char(fieldnames(trainData)));
        fTest  = testData.(char(fieldnames(testData)));

        userData.(sprintf('user%d', user)).fday = fTrain;
        userData.(sprintf('user%d', user)).mday = fTest;

        fprintf('User %02d loaded: FDay [%d x %d], MDay [%d x %d]\n', ...
            user, size(fTrain,1), size(fTrain,2), size(fTest,1), size(fTest,2));
    else
        fprintf('Missing data files for user %02d\n', user);
        fprintf('  Expected:\n    %s\n    %s\n', trainFile, testFile);
    end
end

%% ================== COSINE SIMILARITY & VISUALISATION ==================
fprintf('\nAnalyzing FDay–MDay cosine similarities (subsampled)...\n');

for user = userRange_min:userRange_max
    userField = sprintf('user%d', user);

    if ~isfield(userData, userField)
        fprintf('No data structure for user %02d – skipping.\n', user);
        continue;
    end

    fday_data = userData.(userField).fday;   % [Nf x D]
    mday_data = userData.(userField).mday;   % [Nm x D]

    % --------- 1. Subsample to avoid memory explosion ----------
    Nf = size(fday_data, 1);
    Nm = size(mday_data, 1);

    % choose how many samples to keep in each day (up to MAX_SAMPLES_PER_USER)
    nf_keep = min(Nf, MAX_SAMPLES_PER_USER);
    nm_keep = min(Nm, MAX_SAMPLES_PER_USER);

    % use evenly spaced indices so we cover whole walk, not just first block
    idxF = round(linspace(1, Nf, nf_keep));
    idxM = round(linspace(1, Nm, nm_keep));

    F = fday_data(idxF, :);   % [nf_keep x D]
    M = mday_data(idxM, :);   % [nm_keep x D]

    % --------- 2. Normalise rows and compute cosine similarity ------------
    % Cosine similarity between all pairs: S = F * M' ./ (||F|| * ||M||)
    % First compute row norms
    nF = sqrt(sum(F.^2, 2));   % [nf_keep x 1]
    nM = sqrt(sum(M.^2, 2));   % [nm_keep x 1]

    % Avoid division by zero: replace 0 norms with eps
    nF(nF == 0) = eps;
    nM(nM == 0) = eps;

    % Compute dot products [nf_keep x nm_keep]
    dotProds = F * M.';

    % Denominator matrix = outer product of norms
    denom = nF * nM.';         % [nf_keep x nm_keep]

    similarity_matrix = dotProds ./ denom;

    % Limit numeric issues
    similarity_matrix(~isfinite(similarity_matrix)) = 0;

    % Clip to [0,1] since cosine of gait features should be mostly non-negative
    similarity_matrix = max(min(similarity_matrix, 1), -1);

    % --------- 3. Plot: 3D scatter (spheres) + heatmap -------------------
    figure('Name', sprintf('User %02d – FDay vs MDay Cosine Similarity', user), ...
           'Position', [100 100 1200 500]);

    % (a) 3D scatter with spheres
    subplot(1,2,1);
    hold on;
    [XF, YM] = meshgrid(1:nm_keep, 1:nf_keep);  % note: X=column index, Y=row index

    simVec  = similarity_matrix(:);
    XF_vec  = XF(:);
    YM_vec  = YM(:);

    % plot only highest similarities to keep figure readable
    mask = simVec > PLOT_SIM_THRESHOLD;
    scatter3(XF_vec(mask), YM_vec(mask), simVec(mask), ...
             200 * simVec(mask), ...     % marker size
             simVec(mask), ...           % color
             'filled', ...
             'MarkerFaceAlpha', 0.6);

    hold off;
    colormap('jet');
    colorbar;
    title(sprintf('User %02d: FDay–MDay Similarity (Subsampled Spheres)', user));
    xlabel('MDay Sample Index (subsampled)');
    ylabel('FDay Sample Index (subsampled)');
    zlabel('Cosine Similarity');
    grid on;
    view(45, 40);

    % (b) Heatmap of full subsampled similarity matrix
    subplot(1,2,2);
    imagesc(similarity_matrix);
    colormap('jet');
    colorbar;
    axis square;
    title(sprintf('User %02d: FDay–MDay Similarity Heatmap', user));
    xlabel('MDay Sample Index (subsampled)');
    ylabel('FDay Sample Index (subsampled)');

    % --------- 4. Print some simple stats ---------------------
    diagLen = min(size(similarity_matrix,1), size(similarity_matrix,2));
    diagonal_sim = diag(similarity_matrix(1:diagLen, 1:diagLen));

    fprintf('User %02d – Diagonal mean: %.4f, Overall mean: %.4f, Max: %.4f\n', ...
        user, mean(diagonal_sim), mean(similarity_matrix(:)), max(similarity_matrix(:)));
end

fprintf('\nCosine similarity visualisation complete (subsampled).\n');
