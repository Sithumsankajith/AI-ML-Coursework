%% cosine_3d.m  –  Cosine similarity between FDay and MDay (FULL DATA + SAFE VISUAL)
clear all;
close all;
clc;

rng(100);  % For reproducibility

% User range
userRange_min = 1;
userRange_max = 10;

% File patterns
filePatternsTrain = 'Acc_TimeD_FreqD_FDay';
filePatternsTest  = 'Acc_TimeD_FreqD_MDay';

% Folder where your MAT files are stored
dataDir = 'dataset';   % <-- keep this as in your other scripts

% Max grid size for 3D visualization (to avoid N^2 explosion)
maxGridSize = 3000;     % 200 x 200 = 40,000 points (safe)
anyPlotted  = false;

fprintf('Loading data for each user...\n');

% Initialize storage datasets
userData = struct('trainFeatures', [], 'testFeatures', []);
userData = repmat(userData, 1, userRange_max);

% ---------- LOAD DATA ----------
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

% ---------- COSINE SIMILARITY ANALYSIS ----------
for user = userRange_min:userRange_max
    
    if isempty(userData(user).trainFeatures) || isempty(userData(user).testFeatures)
        fprintf('Skipping user %d - no data\n', user);
        continue;
    end
    
    anyPlotted = true;
    
    fday_full = userData(user).trainFeatures;
    mday_full = userData(user).testFeatures;
    
    % Ensure same feature dimension
    if size(fday_full,2) ~= size(mday_full,2)
        warning('User %d: train and test feature dimensions differ, skipping', user);
        continue;
    end
    
    % Use all matching samples
    numSamplesFull = min(size(fday_full,1), size(mday_full,1));
    fday_full = fday_full(1:numSamplesFull, :);
    mday_full = mday_full(1:numSamplesFull, :);
    
    [numSamplesFull, numFeatures] = size(fday_full);
    fprintf('\nUser %d: FULL data = %d samples x %d features\n', ...
            user, numSamplesFull, numFeatures);
    
    %% 1) FULL-DATA DIAGONAL COSINE SIMILARITY (uses ALL samples)
    %    Compare sample i (FDay) with sample i (MDay)
    normsF = sqrt(sum(fday_full.^2, 2));
    normsM = sqrt(sum(mday_full.^2, 2));
    dotDiag = sum(fday_full .* mday_full, 2);
    
    diagCosSim = dotDiag ./ (normsF .* normsM + eps);
    
    meanDiagSim = mean(diagCosSim);
    stdDiagSim  = std(diagCosSim);
    
    fprintf('  Diagonal cosine similarity (all %d samples): mean = %.4f, std = %.4f\n', ...
            numSamplesFull, meanDiagSim, stdDiagSim);
    
    % Optional: 2D plot of diagonal similarity
    figure('Name', sprintf('User %d Diagonal Cosine Similarity', user));
    plot(1:numSamplesFull, diagCosSim);
    xlabel('Sample index');
    ylabel('Cosine similarity (FDay vs MDay, same index)');
    title(sprintf('User %d – Diagonal Cosine Similarity\nMean = %.3f, Std = %.3f', ...
        user, meanDiagSim, stdDiagSim));
    grid on;
    
    %% 2) SUBSAMPLED 3D COSINE SIMILARITY GRID (for visualization only)
    %    Here we build at most maxGridSize x maxGridSize = 200x200 matrix,
    %    so no out-of-memory, but still representative.
    
    gridSize = min(numSamplesFull, maxGridSize);
    
    % Choose indices spread over the whole walk (not just the first 200)
    idx = round(linspace(1, numSamplesFull, gridSize));
    
    A = fday_full(idx, :);   % FDay subset
    B = mday_full(idx, :);   % MDay subset
    
    % Compute cosine similarity grid using matrix multiplication
    normsA = sqrt(sum(A.^2, 2));      % gridSize x 1
    normsB = sqrt(sum(B.^2, 2));      % gridSize x 1
    
    numer = A * B.';                  % gridSize x gridSize
    denom = normsA * normsB.';        % gridSize x gridSize
    
    cosGrid = numer ./ (denom + eps); % element-wise
    
    % 3D scatter plot of this grid
    [X, Y] = meshgrid(1:gridSize, 1:gridSize);
    
    figure('Name', sprintf('User %d Point-wise Similarity (Subsampled)', user));
    scatter3(X(:), Y(:), cosGrid(:), 10, cosGrid(:), 'filled');
    
    colormap(jet);
    c = colorbar;
    c.Label.String = 'Cosine Similarity';
    clim([0 1]);
    
    xlabel('FDay Sample Index (subsampled)');
    ylabel('MDay Sample Index (subsampled)');
    zlabel('Cosine Similarity');
    title(sprintf(['User %d – Point-wise Cosine Similarity (Subsampled %d of %d)\n' ...
                   'Diagonal mean (full data) = %.3f'], ...
                   user, gridSize, numSamplesFull, meanDiagSim));
    
    grid on;
    rotate3d on;
    view(45, 30);
    xlim([1 gridSize]);
    ylim([1 gridSize]);
    zlim([0 1]);
end

if ~anyPlotted
    warning(['No valid users were loaded. Check that your .mat files are in the "dataset" folder ' ...
             'and named Uxx_Acc_TimeD_FreqD_FDay/MDay.mat']);
end

fprintf('\nCosine similarity analysis complete (FULL DATA for stats, SAFE for plots).\n');
