%% CSV to MAT Pre-processing Script (v2)
% Reads CSVs like 'U1NW_FD.csv'
% Saves MATs like 'U01_Acc_TimeD_FreqD_FDay.mat' for the benchmark script.

clear all; close all; clc;

fprintf('Starting pre-processing: Converting CSV to MAT...\n');

% --- 1. USER MUST CONFIGURE THIS SECTION ---

% Define the column indices from your CSV files.
% !! YOU MUST CHANGE THESE NUMBERS to match your CSVs !!
% Example: If Time Domain features are columns 1 to 50
timeD_cols = 1:3;     
% Example: If Frequency Domain features are columns 51 to 100
freqD_cols = 4:7;   

% Define the feature sets based on the columns above
featureSetDefinitions = {
    % Set 1 Name         % Columns to use
    {'Acc_TimeD_FreqD',   [timeD_cols, freqD_cols]}, ...
    % Set 2 Name
    {'Acc_TimeD',         timeD_cols}, ...
    % Set 3 Name
    {'Acc_FreqD',         freqD_cols}
};

% Define users
userRange = 1:10;

% --- Naming convention for SOURCE CSV files ---
% (e.g., U1NW_FD.csv)
csvUserPrefix = 'U';
csvUserSuffix = 'NW';
csvDataTypes = {'FD', 'MD'}; % Suffix for CSV files

% --- Naming convention for OUTPUT MAT files (for benchmark.m) ---
% (e.g., U01_Acc_TimeD_FreqD_FDay.mat)
matUserPrefix = 'U';
matUserSuffix = ''; % None
matUserFormat = '%02d'; % e.g., '01', '02'
matDataTypes = {'FDay', 'MDay'}; % Suffix for .mat files

% Define source and destination folders
sourceCsvFolder = 'dataset_csv'; % Folder with your original CSVs
destMatFolder = 'dataset';       % Folder for the output .mat files

% --- End of Configuration ---


% Create the destination folder if it doesn't exist
if ~exist(destMatFolder, 'dir')
    mkdir(destMatFolder);
end

% --- 2. Main Processing Loop ---
fprintf('Scanning for %d users...\n', length(userRange));

for user_idx = 1:length(userRange)
    user = userRange(user_idx);
    
    % --- Construct SOURCE (CSV) names ---
    % e.g., 'U' + 1 + 'NW' = 'U1NW'
    csvUserStr = sprintf('%s%d%s', csvUserPrefix, user, csvUserSuffix);
    
    % --- Construct DESTINATION (MAT) names ---
    % e.g., 'U' + '01' + '' = 'U01'
    matUserStr = sprintf('%s%s%s', matUserPrefix, sprintf(matUserFormat, user), matUserSuffix);

    for dt_idx = 1:length(csvDataTypes)
        
        csvDataType = csvDataTypes{dt_idx}; % e.g., 'FD'
        matDataType = matDataTypes{dt_idx}; % e.g., 'FDay'
        
        % Construct the source CSV file name (e.g., U1NW_FD.csv)
        csvFileName = sprintf('%s_%s.csv', csvUserStr, csvDataType);
        csvFilePath = fullfile(sourceCsvFolder, csvFileName);
        
        if ~exist(csvFilePath, 'file')
            fprintf('WARNING: CSV file not found, skipping: %s\n', csvFilePath);
            continue; % Skip to the next file
        end
        
        % Read the entire CSV file
        fprintf('Reading: %s\n', csvFileName);
        try
            % Use readmatrix if your CSVs are all numbers, no text headers
            all_data_from_csv = readmatrix(csvFilePath);
            
            % ---
            % NOTE: If your CSV has a header row (e.g., "Feature1", "Feature2")
            % you should use readtable instead:
            %
            % T = readtable(csvFilePath);
            % all_data_from_csv = T{:,:}; % Convert table to matrix
            % ---

        catch ME
            fprintf('Error reading %s: %s\n', csvFileName, ME.message);
            continue;
        end
        
        % Now, create and save the 3 feature sets from this one CSV
        for fs_idx = 1:length(featureSetDefinitions)
            
            setName = featureSetDefinitions{fs_idx}{1};
            setCols = featureSetDefinitions{fs_idx}{2};
            
            % Create the final feature set name (e.g., 'Acc_TimeD_FreqD_FDay')
            % This name is for the .mat file, so it uses 'matDataType'
            finalFeatureName = [setName, '_', matDataType];
            
            % Construct the destination .mat file name
            % This uses 'matUserStr' (e.g., U01)
            matFileName = sprintf('%s_%s.mat', matUserStr, finalFeatureName);
            matFilePath = fullfile(destMatFolder, matFileName);
            
            % Extract the feature columns from the data
            try
                featureMatrix = all_data_from_csv(:, setCols);
            catch ME
                fprintf('Error: Could not extract columns for %s. Check your column indices. %s\n', setName, ME.message);
                continue;
            end
            
            % --- Save the .mat file ---
            % We create a temporary struct to save the variable with a 
            % specific name inside the .mat file, as required by 
            % your benchmark script's loading method.
            
            varName = [matUserStr, '_', finalFeatureName, '_data'];
            dataToSave = struct();
            dataToSave.(varName) = featureMatrix;
            
            fprintf('  -> Saving as: %s\n', matFileName);
            save(matFilePath, '-struct', 'dataToSave');
            
        end % end loop over feature sets
    end % end loop over data types (FD/MD)
end % end loop over users

fprintf('\nAll CSV files processed and .mat files created in /dataset/ folder.\n');
fprintf('You can now run your benchmark script (run_benchmark.m).\n');