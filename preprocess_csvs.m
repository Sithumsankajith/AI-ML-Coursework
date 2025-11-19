%% CSV to MAT Pre-processing Script (v3)
% Reads RAW CSVs like 'U1NW_FD.csv' containing:
%   [timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
% Computes sliding-window Time Domain + Frequency Domain features
% (for accelerometer only) and saves MATs like:
%   'U01_Acc_TimeD_FreqD_FDay.mat',
%   'U01_Acc_TimeD_FDay.mat',
%   'U01_Acc_FreqD_FDay.mat'
%
% Each MAT contains ONE variable named e.g.:
%   U01_Acc_TimeD_FreqD_FDay_data   (samples x features)

clear all; close all; clc;

fprintf('Starting pre-processing: Converting RAW CSV to MAT (features)...\n');

%% ------------------------------------------------------------------------
% 1. USER CONFIGURATION
% -------------------------------------------------------------------------

% ---- Sliding window settings ----
fs      = 31;      % sampling frequency (Hz), approx 30–32
winSize = 128;     % samples per window (≈4 s)
overlap = 0.5;     % 50%% overlap
step    = round(winSize * (1 - overlap));

% ---- Users and file naming ----
userRange     = 1:10;

% SOURCE CSV naming: e.g. 'U1NW_FD.csv'
csvUserPrefix = 'U';
csvUserSuffix = 'NW';
csvDataTypes  = {'FD', 'MD'};     % FD = first day, MD = second day

% OUTPUT MAT naming: e.g. 'U01_Acc_TimeD_FreqD_FDay.mat'
matUserPrefix = 'U';
matUserSuffix = '';               % none
matUserFormat = '%02d';           % -> '01', '02', ...
matDataTypes  = {'FDay', 'MDay'}; % correspond to FD/MD

% ---- Folders ----
sourceCsvFolder = 'dataset_csv';  % folder with original RAW CSVs
destMatFolder   = 'dataset';      % folder for the output .mat feature files

% Create output folder if needed
if ~exist(destMatFolder, 'dir')
    mkdir(destMatFolder);
end

fprintf('Sliding window: %d samples, %.0f%%%% overlap, fs = %.1f Hz\n', ...
        winSize, overlap*100, fs);
fprintf('Users: %s\n', mat2str(userRange));

%% ------------------------------------------------------------------------
% 2. Main Processing Loop
% -------------------------------------------------------------------------

for user = userRange
    
    % Build user strings
    csvUserStr = sprintf('%s%d%s',      csvUserPrefix, user, csvUserSuffix);       % e.g. U1NW
    matUserStr = sprintf('%s%s%s', matUserPrefix, sprintf(matUserFormat, user), matUserSuffix); % e.g. U01
    
    for dt_idx = 1:length(csvDataTypes)
        
        csvDataType = csvDataTypes{dt_idx};   % 'FD' or 'MD'
        matDataType = matDataTypes{dt_idx};   % 'FDay' or 'MDay'
        
        % Source CSV file name & path
        csvFileName = sprintf('%s_%s.csv', csvUserStr, csvDataType);
        csvFilePath = fullfile(sourceCsvFolder, csvFileName);
        
        if ~exist(csvFilePath, 'file')
            fprintf('WARNING: CSV file not found, skipping: %s\n', csvFilePath);
            continue;
        end
        
        fprintf('\nUser %02d, %s: Reading %s\n', user, matDataType, csvFileName);
        
        % ---- Read raw CSV ----
        try
            raw = readmatrix(csvFilePath); % numeric, no header
        catch ME
            fprintf('Error reading %s: %s\n', csvFileName, ME.message);
            continue;
        end
        
        if size(raw,2) < 4
            fprintf('ERROR: %s has <4 columns. Expected: [time,acc_x,acc_y,acc_z,...]\n', csvFileName);
            continue;
        end
        
        % Columns: 1 = time, 2:4 = accelerometer
        acc = raw(:, 2:4);   % [acc_x, acc_y, acc_z]
        N   = size(acc,1);
        
        if N < winSize
            fprintf('Not enough samples (%d) for one window (%d). Skipping.\n', N, winSize);
            continue;
        end
        
        % Storage for this user & day
        Acc_TimeD = [];  % time-domain only
        Acc_FreqD = [];  % frequency-domain only
        
        % ---- Sliding window over samples ----
        for startIdx = 1:step:(N - winSize + 1)
            endIdx = startIdx + winSize - 1;
            segAcc = acc(startIdx:endIdx, :); % window: winSize x 3
            
            td_row = [];
            fd_row = [];
            
            % For each accelerometer axis (x,y,z)
            for axisIdx = 1:3
                x = segAcc(:, axisIdx);
                
                td = extractTimeFeatures(x);
                fd = extractFreqFeatures(x, fs);
                
                td_row = [td_row, td];
                fd_row = [fd_row, fd];
            end
            
            Acc_TimeD = [Acc_TimeD; td_row];
            Acc_FreqD = [Acc_FreqD; fd_row];
        end
        
        % Combined feature set: TimeD + FreqD
        Acc_TimeD_FreqD = [Acc_TimeD, Acc_FreqD];
        
        % -----------------------------------------------------------------
        % 3. Save three MAT files:
        %    U01_Acc_TimeD_FreqD_FDay.mat
        %    U01_Acc_TimeD_FDay.mat
        %    U01_Acc_FreqD_FDay.mat
        % with variables:
        %    U01_Acc_TimeD_FreqD_FDay_data, etc.
        % -----------------------------------------------------------------
        
        % --- 3a. TimeD + FreqD ---
        finalFeatureName = ['Acc_TimeD_FreqD_', matDataType];  % e.g. 'Acc_TimeD_FreqD_FDay'
        matFileName      = sprintf('%s_%s.mat', matUserStr, finalFeatureName);
        matFilePath      = fullfile(destMatFolder, matFileName);
        
        varName          = [matUserStr, '_', finalFeatureName, '_data']; % e.g. U01_Acc_TimeD_FreqD_FDay_data
        dataToSave       = struct();
        dataToSave.(varName) = Acc_TimeD_FreqD;
        fprintf('  -> Saving %s\n', matFileName);
        save(matFilePath, '-struct', 'dataToSave');
        
        % --- 3b. TimeD only ---
        finalFeatureName = ['Acc_TimeD_', matDataType];        % e.g. 'Acc_TimeD_FDay'
        matFileName      = sprintf('%s_%s.mat', matUserStr, finalFeatureName);
        matFilePath      = fullfile(destMatFolder, matFileName);
        
        varName          = [matUserStr, '_', finalFeatureName, '_data']; % e.g. U01_Acc_TimeD_FDay_data
        dataToSave       = struct();
        dataToSave.(varName) = Acc_TimeD;
        fprintf('  -> Saving %s\n', matFileName);
        save(matFilePath, '-struct', 'dataToSave');
        
        % --- 3c. FreqD only ---
        finalFeatureName = ['Acc_FreqD_', matDataType];        % e.g. 'Acc_FreqD_FDay'
        matFileName      = sprintf('%s_%s.mat', matUserStr, finalFeatureName);
        matFilePath      = fullfile(destMatFolder, matFileName);
        
        varName          = [matUserStr, '_', finalFeatureName, '_data']; % e.g. U01_Acc_FreqD_FDay_data
        dataToSave       = struct();
        dataToSave.(varName) = Acc_FreqD;
        fprintf('  -> Saving %s\n', matFileName);
        save(matFilePath, '-struct', 'dataToSave');
        
    end % FD / MD
end % users

fprintf('\nAll users processed. MAT feature files are in folder: %s\n', destMatFolder);

%% ------------------------------------------------------------------------
%  Local functions: time-domain & frequency-domain features
% -------------------------------------------------------------------------

function td = extractTimeFeatures(x)
% Time-domain features for a 1-D signal x
    x = x(:);
    n = numel(x);

    meanX  = mean(x);
    stdX   = std(x);
    varX   = var(x);
    medX   = median(x);
    minX   = min(x);
    maxX   = max(x);
    rangeX = maxX - minX;
    iqrX   = iqr(x);                 % needs Statistics toolbox
    rmsX   = sqrt(mean(x.^2));
    smaX   = mean(abs(x));
    
    if n > 1
        zc = sum(diff(sign(x)) ~= 0) / (n - 1);  % zero-crossing rate
    else
        zc = 0;
    end
    
    if stdX > 0
        skewX = mean(((x - meanX)/stdX).^3);
        kurtX = mean(((x - meanX)/stdX).^4) - 3; % excess kurtosis
    else
        skewX = 0;
        kurtX = 0;
    end
    
    td = [meanX, stdX, varX, medX, minX, maxX, ...
          rangeX, iqrX, rmsX, smaX, zc, skewX, kurtX];
end

function fd = extractFreqFeatures(x, fs)
% Frequency-domain features for a 1-D signal x
    x = x(:);
    N = numel(x);
    
    X = fft(x);
    K = floor(N/2) + 1;        % one-sided spectrum
    X = X(1:K);
    mag   = abs(X);
    freqs = (0:K-1)' * (fs / N);
    
    total = sum(mag) + eps;
    p     = mag / total;
    
    [~, idxMax] = max(mag);
    domFreq     = freqs(idxMax);
    specEnergy  = sum(mag.^2);
    specEntropy = -sum(p .* log(p + eps));
    peakAmp     = max(mag);
    meanFreq    = sum(freqs .* mag) / total;
    freqVar     = sum(((freqs - meanFreq).^2) .* mag) / total;
    
    fd = [domFreq, specEnergy, specEntropy, ...
          peakAmp, meanFreq, freqVar];
end
