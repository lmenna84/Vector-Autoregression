%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% EXAMPLES FOR nonrecursiveSRSVAR
%% Author: Lorenzo Menna
%% This script demonstrates all features of the nonrecursiveSRSVAR function
%% (Structural VAR with short-run non-recursive zero restrictions)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all; clc;

%% ========================================================================
%% DATA SIMULATION
%% ========================================================================
% We simulate a 3-variable VAR(2) system with known parameters

rng(12345);  % Set seed for reproducibility

% Parameters
K = 3;        % Number of variables
T = 200;      % Sample size
nlags = 2;    % Number of lags

% True VAR coefficient matrices (stable system)
A1 = [0.5  0.1  0.0;
      0.1  0.6  0.1;
      0.0  0.1  0.4];

A2 = [0.2  0.0  0.1;
      0.0  0.1  0.0;
      0.1  0.0  0.2];

nu = [0.5; 1.0; 0.2];  % Constants

% True structural impact matrix (non-recursive pattern)
% Matching idmat1: [1 1 0; 1 1 0; 0 1 1]
B_true = [1.0  0.4  0.0;
          0.4  0.9  0.0;
          0.0  0.3  0.8];

% Generate structural shocks (orthogonal)
epsilon = randn(K, T);

% Generate VAR data
data = zeros(T, K);
data(1:nlags, :) = randn(nlags, K);  % Initial values

for t = (nlags+1):T
    structural_shock = B_true * epsilon(:, t);
    data(t, :) = nu' + data(t-1, :) * A1' + data(t-2, :) * A2' + structural_shock';
end

disp('========================================');
disp('Data simulation complete');
disp(['Sample size: ' num2str(T)]);
disp(['Number of variables: ' num2str(K)]);
disp('True non-recursive impact matrix B:');
disp(B_true);
disp('========================================');
fprintf('\n');

%% ========================================================================
%% EXAMPLE 1: Basic non-recursive SVAR
%% ========================================================================
disp('========================================');
disp('EXAMPLE 1: Non-recursive SVAR with specific zero pattern');
disp('========================================');

% Define variable names
var_names = {'Output', 'Inflation', 'InterestRate'};

% Define identification matrix
% idmat(i,j) = 1 means shock j affects variable i contemporaneously
idmat1 = [1  1  0;    % Output affected by shocks 1 and 2
          1  1  0;    % Inflation affected by shocks 1 and 2
          0  1  1];   % Interest rate affected by shocks 2 and 3

% Check exact identification
disp('Identification matrix:');
disp(idmat1);
fprintf('Number of non-zero elements: %d\n', sum(idmat1(:)));
fprintf('Required for exact identification: %d\n', K*(K+1)/2);
if sum(idmat1(:))==K*(K+1)/2
    fprintf('Status: EXACTLY IDENTIFIED\n');
else
    fprintf('Status: ERROR - NOT EXACTLY IDENTIFIED\n');
end
fprintf('\n');

EQ1 = nonrecursiveSRSVAR(data, 2, idmat1, var_names, 40, 1000, true);
close all;

disp('Estimated impact matrix P:');
disp(EQ1.P);
disp('True impact matrix B:');
disp(B_true);
fprintf('\n');

% Display IRF
disp('IRF: Inflation response to Output shock (first 10 periods):');
disp('Period    95% CI    IRF    5% CI');
for i = 1:10
    fprintf('%3d    %7.4f  %7.4f  %7.4f\n', i, ...
            EQ1.Inflation_Output(1,i), EQ1.Inflation_Output(2,i), EQ1.Inflation_Output(3,i));
end
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 2: Different non-recursive identification pattern
%% ========================================================================
disp('========================================');
disp('EXAMPLE 2: Alternative non-recursive identification');
disp('========================================');

% Different pattern with 6 non-zero elements
idmat2 = [1  0  1;    % Output affected by shocks 1 and 3
          1  1  0;    % Inflation affected by shocks 1 and 2
          1  0  1];   % Interest rate affected by shocks 1 and 3

disp('Identification matrix:');
disp(idmat2);
fprintf('Number of non-zero elements: %d (exactly identified)\n', sum(idmat2(:)));
fprintf('\n');

EQ2 = nonrecursiveSRSVAR(data, 2, idmat2, var_names, 40, 1000, true);
close all;

disp('Estimated impact matrix P:');
disp(EQ2.P);
fprintf('Note: This pattern has shock 3 affecting output and interest rate\n');
fprintf('      but inflation is isolated from shock 3\n');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 3: Comparison with recursive (Cholesky) identification
%% ========================================================================
disp('========================================');
disp('EXAMPLE 3: Non-recursive vs Recursive (Cholesky) comparison');
disp('========================================');

% Recursive identification (lower triangular)
idmat_recursive = [1  0  0;
                   1  1  0;
                   1  1  1];

disp('Recursive (Cholesky) identification matrix:');
disp(idmat_recursive);
fprintf('Number of non-zero elements: %d\n', sum(idmat_recursive(:)));
fprintf('\n');

EQ3_nonrec = nonrecursiveSRSVAR(data, 2, idmat_recursive, var_names, 40, 1000, true);
close all;
EQ3_chol = srSVAR(data, 2, var_names, 40, 1000, true);
close all;

disp('Comparison of impact matrices:');
disp('Non-recursive method (with recursive pattern):');
disp(EQ3_nonrec.P);
disp('Cholesky decomposition:');
disp(EQ3_chol.P);
disp('Difference (should be near zero):');
disp(abs(EQ3_nonrec.P - EQ3_chol.P));
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 4: Custom shock sizes
%% ========================================================================
disp('========================================');
disp('EXAMPLE 4: Non-recursive SVAR with custom shock sizes');
disp('2 standard deviation shocks');
disp('========================================');

shock_size = 2 * diag(EQ1.P);

EQ4 = nonrecursiveSRSVAR(data, 2, idmat1, var_names, 40, 1000, true, [], [], [], shock_size);
close all;

disp('Comparison of IRFs (Inflation response to Output shock at horizon 5):');
fprintf('1 std shock:  %.4f [%.4f, %.4f]\n', ...
        EQ1.Inflation_Output(2,5), EQ1.Inflation_Output(3,5), EQ1.Inflation_Output(1,5));
fprintf('2 std shock:  %.4f [%.4f, %.4f]\n', ...
        EQ4.Inflation_Output(2,5), EQ4.Inflation_Output(3,5), EQ4.Inflation_Output(1,5));
fprintf('Ratio:        %.4f (should be ~2.0)\n', ...
        EQ4.Inflation_Output(2,5) / EQ1.Inflation_Output(2,5));
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 5: Different confidence levels
%% ========================================================================
disp('========================================');
disp('EXAMPLE 5: Non-recursive SVAR with different confidence levels');
disp('========================================');

EQ5a = nonrecursiveSRSVAR(data, 2, idmat1, var_names, 40, 1000, true, [], [], [], [], 5);   % 90% CI
close all;
EQ5b = nonrecursiveSRSVAR(data, 2, idmat1, var_names, 40, 1000, true, [], [], [], [], 16);  % 68% CI
close all;

disp('Comparison of confidence intervals (Output response to Output shock at horizon 5):');
fprintf('90%% CI: [%.4f, %.4f]  Width: %.4f\n', ...
        EQ5a.Output_Output(3,5), EQ5a.Output_Output(1,5), ...
        EQ5a.Output_Output(1,5) - EQ5a.Output_Output(3,5));
fprintf('68%% CI: [%.4f, %.4f]  Width: %.4f\n', ...
        EQ5b.Output_Output(3,5), EQ5b.Output_Output(1,5), ...
        EQ5b.Output_Output(1,5) - EQ5b.Output_Output(3,5));
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 6: With seasonal dummies
%% ========================================================================
disp('========================================');
disp('EXAMPLE 6: Non-recursive SVAR with quarterly seasonal dummies');
disp('========================================');

% Simulate quarterly data with seasonal pattern
T_q = 120;
data_seasonal = zeros(T_q, K);
data_seasonal(1:nlags, :) = randn(nlags, K);

seasonal_pattern = [0.5; -0.3; 0.2; -0.4];

for t = (nlags+1):T_q
    quarter = mod(t-1, 4) + 1;
    seasonal_effect = seasonal_pattern(quarter) * [1; 0.5; 0.3]';
    structural_shock = B_true * randn(K, 1);
    
    data_seasonal(t, :) = nu' + data_seasonal(t-1, :) * A1' + ...
                          data_seasonal(t-2, :) * A2' + seasonal_effect + structural_shock';
end

EQ6 = nonrecursiveSRSVAR(data_seasonal, 2, idmat1, var_names, 40, 1000, true, [], 'quarter');
close all;

disp('Non-recursive SVAR with quarterly dummies estimated successfully');
disp('IRF: Output response to Output shock (periods 1-5):');
for i = 1:5
    fprintf('Period %d:  %.4f [%.4f, %.4f]\n', i, ...
            EQ6.Output_Output(2,i), EQ6.Output_Output(3,i), EQ6.Output_Output(1,i));
end
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 7: With exogenous variables
%% ========================================================================
disp('========================================');
disp('EXAMPLE 7: Non-recursive SVAR with exogenous crisis dummy');
disp('Crisis period: observations 80-90');
disp('========================================');

% Create crisis dummy
exog = zeros(T, 1);
exog(80:90) = 1;

% Add crisis effect to data
data_crisis = data;
crisis_effect = B_true * [-2; -1.5; -1];
for t = 80:90
    data_crisis(t, :) = data_crisis(t, :) + crisis_effect';
end

EQ7 = nonrecursiveSRSVAR(data_crisis, 2, idmat1, var_names, 40, 1000, true, [], [], exog);
close all;

disp('Non-recursive SVAR with crisis dummy estimated');
disp('Structural shocks during crisis period (observations 80-85):');
disp('Obs    Output    Inflation    InterestRate');
for i = 80:85
    fprintf('%3d   %7.3f   %7.3f      %7.3f\n', i, ...
            EQ7.struc(1,i-nlags), EQ7.struc(2,i-nlags), EQ7.struc(3,i-nlags));
end
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 8: Restricted reduced-form VAR
%% ========================================================================
disp('========================================');
disp('EXAMPLE 8: Non-recursive SVAR with restricted reduced-form VAR');
disp('Interest rate does not affect output or inflation in lags');
disp('========================================');

lr = [1 1 0;   % Output affected by output, inflation only
      1 1 0;   % Inflation affected by output, inflation only
      1 1 1];  % Interest rate affected by all variables

EQ8 = nonrecursiveSRSVAR(data, 2, idmat1, var_names, 40, 1000, true, lr);
close all;

disp('Non-recursive identification + reduced-form restrictions');
disp('Impact matrix P (non-recursive):');
disp(EQ8.P);
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 9: Structural shocks analysis
%% ========================================================================
disp('========================================');
disp('EXAMPLE 9: Analyzing structural shocks');
disp('========================================');

% Plot structural shocks
figure('Position', [100 100 1200 800]);

for i = 1:K
    subplot(K, 1, i);
    plot(1:T-nlags, EQ1.struc(i,:), 'k-', 'LineWidth', 1);
    yline(0, 'r--', 'LineWidth', 0.5);
    xlabel('Time');
    ylabel('Shock');
    title([var_names{i} ' structural shock']);
    grid on;
end

sgtitle('Estimated Structural Shocks (Non-recursive Identification)');
fprintf('Structural shocks plotted (Figure 1)\n');
fprintf('\n');

% Summary statistics
disp('Summary statistics of structural shocks:');
disp('Shock           Mean      Std Dev    Min       Max');
for i = 1:K
    fprintf('%-15s %.4f    %.4f     %.4f    %.4f\n', ...
            var_names{i}, mean(EQ1.struc(i,:)), std(EQ1.struc(i,:)), ...
            min(EQ1.struc(i,:)), max(EQ1.struc(i,:)));
end
fprintf('\n');

% Correlation of structural shocks
disp('Correlation matrix of structural shocks:');
disp(corr(EQ1.struc'));
fprintf('Note: Shocks should be approximately uncorrelated\n');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 10: Variance decomposition
%% ========================================================================
disp('========================================');
disp('EXAMPLE 10: Forecast error variance decomposition');
disp('========================================');

horizons = [1, 4, 10, 20, 40];

disp('Variance decomposition of Inflation:');
disp('Horizon  Output    Inflation  InterestRate');
for h = horizons
    fprintf('%3d      %6.2f%%   %6.2f%%     %6.2f%%\n', h, ...
            EQ1.vardecshock_Inflation(1, h), ...
            EQ1.vardecshock_Inflation(2, h), ...
            EQ1.vardecshock_Inflation(3, h));
end
fprintf('\n');

% Plot variance decomposition
figure('Position', [100 100 1200 400]);
for i = 1:K
    subplot(1, K, i);
    
    if i == 1
        vardec = EQ1.vardecshock_Output;
    elseif i == 2
        vardec = EQ1.vardecshock_Inflation;
    else
        vardec = EQ1.vardecshock_InterestRate;
    end
    
    hold on;
    plot(1:40, vardec(1,:), 'b-', 'LineWidth', 2, 'DisplayName', 'Output');
    plot(1:40, vardec(2,:), 'r-', 'LineWidth', 2, 'DisplayName', 'Inflation');
    plot(1:40, vardec(3,:), 'g-', 'LineWidth', 2, 'DisplayName', 'InterestRate');
    hold off;
    
    xlabel('Horizon');
    ylabel('Percentage');
    title([var_names{i} ' variance decomposition']);
    legend('Location', 'best');
    grid on;
    ylim([0 100]);
end

sgtitle('Forecast Error Variance Decomposition (Non-recursive Identification)');
fprintf('Variance decomposition plot created (Figure 2)\n');
fprintf('\n\n');

%% ========================================================================
disp('========================================');
disp('All examples completed successfully!');
disp('========================================');
disp('Key features demonstrated:');
disp('- Non-recursive zero restrictions (exactly identified)');
disp('- Different identification patterns');
disp('- Comparison with recursive (Cholesky) identification');
disp('- Custom shock sizes and confidence levels');
disp('- Seasonal dummies and exogenous variables');
disp('- Restricted reduced-form VARs');
disp('- Structural shocks recovery and analysis');
disp('- Variance decomposition');