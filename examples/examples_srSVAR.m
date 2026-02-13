%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% EXAMPLES FOR srSVAR
%% Author: Lorenzo Menna
%% This script demonstrates all features of the srSVAR function
%% (Structural VAR with short-run restrictions via Cholesky decomposition)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all; clc;

%% ========================================================================
%% DATA SIMULATION
%% ========================================================================
% We simulate a 3-variable VAR(2) system with known parameters
% This allows us to verify the function recovers the true structural shocks

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

% True structural impact matrix (lower triangular for Cholesky)
B_true = [1.0  0.0  0.0;
          0.3  0.9  0.0;
          0.1  0.2  0.8];

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
disp('========================================');
fprintf('\n');

%% ========================================================================
%% EXAMPLE 1: Basic SVAR with Cholesky identification
%% ========================================================================
disp('========================================');
disp('EXAMPLE 1: Basic SVAR(2) with Cholesky decomposition');
disp('========================================');

% Define variable names
var_names = {'Output', 'Inflation', 'InterestRate'};

EQ1 = srSVAR(data, 2, var_names, 40, 1000, true);
close all

disp('Estimated Cholesky matrix (impact of structural shocks on residuals):');
disp(EQ1.P);
disp('True structural impact matrix:');
disp(B_true);
fprintf('\n');

% Display IRF for Output responding to Output shock
disp('IRF: Output response to Output shock (first 10 periods):');
disp('Period    95% CI    IRF    5% CI');
for i = 1:10
    fprintf('%3d    %7.4f  %7.4f  %7.4f\n', i, ...
            EQ1.Output_Output(1,i), EQ1.Output_Output(2,i), EQ1.Output_Output(3,i));
end
fprintf('\n');

% Variance decomposition at horizon 10
disp('Variance decomposition of Output at horizon 10:');
fprintf('Output shock:        %.2f%%\n', EQ1.vardecshock_Output(1, 10));
fprintf('Inflation shock:     %.2f%%\n', EQ1.vardecshock_Inflation(1, 10));
fprintf('InterestRate shock:  %.2f%%\n', EQ1.vardecshock_InterestRate(1, 10));
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 2: SVAR with custom shock sizes
%% ========================================================================
disp('========================================');
disp('EXAMPLE 2: SVAR with custom shock sizes');
disp('2 standard deviation shocks');
disp('========================================');

% Define shock sizes (2 std dev for each variable)
shock_size = 2 * diag(EQ1.P);  % 2 times the standard deviation

EQ2 = srSVAR(data, 2, var_names, 40, 1000, true, [], [], [], shock_size);
close all

disp('Comparison of IRFs (Output response to Output shock at horizon 5):');
fprintf('1 std shock:  %.4f [%.4f, %.4f]\n', ...
        EQ1.Output_Output(2,5), EQ1.Output_Output(3,5), EQ1.Output_Output(1,5));
fprintf('2 std shock:  %.4f [%.4f, %.4f]\n', ...
        EQ2.Output_Output(2,5), EQ2.Output_Output(3,5), EQ2.Output_Output(1,5));
fprintf('Ratio:        %.4f (should be ~2.0)\n', ...
        EQ2.Output_Output(2,5) / EQ1.Output_Output(2,5));
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 3: SVAR with restrictions on variable interactions
%% ========================================================================
disp('========================================');
disp('EXAMPLE 3: Restricted SVAR');
disp('Restriction: Interest rate does not affect output or inflation');
disp('========================================');

% Restriction matrix
lr = [1 1 0;   % Output affected by output, inflation only
      1 1 0;   % Inflation affected by output, inflation only
      1 1 1];  % Interest rate affected by all variables

EQ3 = srSVAR(data, 2, var_names, 40, 1000, true, lr);
close all

disp('Unrestricted: Effect of lagged interest rate on output:');
disp('         Lag 1    Lag 2');
fprintf('Output:  %7.4f  %7.4f\n', EQ1.P(1,3), EQ1.P(1,3));  % This is from reduced form

disp('Restricted: Coefficients should reflect restriction in reduced form VAR');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 4: SVAR with different confidence levels
%% ========================================================================
disp('========================================');
disp('EXAMPLE 4: SVAR with different confidence levels');
disp('========================================');

% 16% confidence level (68% confidence interval, common in Bayesian analysis)
EQ4 = srSVAR(data, 2, var_names, 40, 1000, true, [], [], [], [], 16);
close all

disp('Comparison of confidence intervals (Output response to Output shock at horizon 5):');
fprintf('90%% CI: [%.4f, %.4f]  Width: %.4f\n', ...
        EQ1.Output_Output(3,5), EQ1.Output_Output(1,5), ...
        EQ1.Output_Output(1,5) - EQ1.Output_Output(3,5));
fprintf('68%% CI: [%.4f, %.4f]  Width: %.4f\n', ...
        EQ4.Output_Output(3,5), EQ4.Output_Output(1,5), ...
        EQ4.Output_Output(1,5) - EQ4.Output_Output(3,5));
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 5: SVAR with seasonal dummies
%% ========================================================================
disp('========================================');
disp('EXAMPLE 5: SVAR with quarterly seasonal dummies');
disp('========================================');

% Simulate quarterly data with seasonal pattern
T_q = 120;  % 30 years of quarterly data
data_seasonal = zeros(T_q, K);
data_seasonal(1:nlags, :) = randn(nlags, K);

% Create seasonal effects
seasonal_pattern = [0.5; -0.3; 0.2; -0.4];

for t = (nlags+1):T_q
    quarter = mod(t-1, 4) + 1;
    seasonal_effect = seasonal_pattern(quarter) * [1; 0.5; 0.3]';
    structural_shock = B_true * randn(K, 1);
    
    data_seasonal(t, :) = nu' + data_seasonal(t-1, :) * A1' + ...
                          data_seasonal(t-2, :) * A2' + seasonal_effect + structural_shock';
end

EQ5 = srSVAR(data_seasonal, 2, var_names, 40, 1000, true, [], 'quarter');
close all

disp('SVAR estimated with quarterly seasonal dummies');
disp('IRF: Output response to Output shock (periods 1-5):');
for i = 1:5
    fprintf('Period %d:  %.4f [%.4f, %.4f]\n', i, ...
            EQ5.Output_Output(2,i), EQ5.Output_Output(3,i), EQ5.Output_Output(1,i));
end
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 6: SVAR with exogenous variables
%% ========================================================================
disp('========================================');
disp('EXAMPLE 6: SVAR with exogenous crisis dummy');
disp('Crisis period: observations 80-90');
disp('========================================');

% Create crisis dummy
exog = zeros(T, 1);
exog(80:90) = 1;

% Add crisis effect to data
data_crisis = data;
crisis_effect = B_true * [-2; -1.5; -1];  % Negative structural shock during crisis
for t = 80:90
    data_crisis(t, :) = data_crisis(t, :) + crisis_effect';
end

% Estimate with exogenous variable
EQ6 = srSVAR(data_crisis, 2, var_names, 40, 1000, true, [], [], exog);
close all

disp('SVAR with crisis dummy estimated successfully');
disp('Structural shocks during crisis period (observations 80-85):');
disp('Obs    Output    Inflation    InterestRate');
for i = 80:85
    fprintf('%3d   %7.3f   %7.3f      %7.3f\n', i, ...
            EQ6.struc(1,i-nlags), EQ6.struc(2,i-nlags), EQ6.struc(3,i-nlags));
end
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 7: Plotting impulse response functions
%% ========================================================================
disp('========================================');
disp('EXAMPLE 7: Plotting IRFs');
disp('========================================');

% Plot IRFs for Output equation
figure('Position', [100 100 1200 400]);
shocks = {'Output', 'Inflation', 'InterestRate'};

for j = 1:K
    subplot(1, K, j);
    
    % Get the IRF data
    if j == 1
        irf_data = EQ1.Output_Output;
    elseif j == 2
        irf_data = EQ1.Output_Inflation;
    else
        irf_data = EQ1.Output_InterestRate;
    end
    
    hold on;
    % Plot confidence bands
    plot(1:40, irf_data(1,:), 'k--', 'LineWidth', 1);
    plot(1:40, irf_data(3,:), 'k--', 'LineWidth', 1);
    % Plot IRF
    plot(1:40, irf_data(2,:), 'b-', 'LineWidth', 2);
    % Zero line
    yline(0, 'k:', 'LineWidth', 0.5);
    
    xlabel('Horizon');
    ylabel('Response');
    title(['Output response to ' shocks{j} ' shock']);
    legend('90% CI', '', 'IRF', 'Location', 'best');
    grid on;
    hold off;
end

sgtitle('Impulse Response Functions: Output equation');

disp('IRF plot created (Figure 1)');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 8: Analyzing structural shocks
%% ========================================================================
disp('========================================');
disp('EXAMPLE 8: Structural shocks analysis');
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

sgtitle('Estimated Structural Shocks');

disp('Structural shocks plotted (Figure 2)');
fprintf('\n');

% Summary statistics of structural shocks
disp('Summary statistics of structural shocks:');
disp('Shock           Mean      Std Dev    Min       Max');
for i = 1:K
    fprintf('%-15s %.4f    %.4f     %.4f    %.4f\n', ...
            var_names{i}, mean(EQ1.struc(i,:)), std(EQ1.struc(i,:)), ...
            min(EQ1.struc(i,:)), max(EQ1.struc(i,:)));
end
fprintf('\n');

% Correlation of structural shocks (should be close to zero)
disp('Correlation matrix of structural shocks (should be ~0):');
disp(corr(EQ1.struc'));
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 9: Variance decomposition analysis
%% ========================================================================
disp('========================================');
disp('EXAMPLE 9: Forecast error variance decomposition');
disp('========================================');

% Variance decomposition at different horizons
horizons = [1, 4, 10, 20, 40];

disp('Variance decomposition of Output:');
disp('Horizon  Output    Inflation  InterestRate');
for h = horizons
    fprintf('%3d      %6.2f%%   %6.2f%%     %6.2f%%\n', h, ...
            EQ1.vardecshock_Output(1, h), ...
            EQ1.vardecshock_Inflation(1, h), ...
            EQ1.vardecshock_InterestRate(1, h));
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

sgtitle('Forecast Error Variance Decomposition');

disp('Variance decomposition plot created (Figure 3)');
fprintf('\n\n');

%% ========================================================================
disp('========================================');
disp('All examples completed successfully!');
disp('========================================');