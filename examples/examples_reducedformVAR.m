%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% EXAMPLES FOR reducedformVAR
%% Author: Lorenzo Menna
%% This script demonstrates all features of the reducedformVAR function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all; clc;

%% ========================================================================
%% DATA SIMULATION
%% ========================================================================
% We simulate a 3-variable VAR(2) system with known parameters
% This allows us to verify the function recovers the true parameters

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

% Variance-covariance matrix of shocks
Sigma_u = [1.0  0.3  0.1;
           0.3  0.8  0.2;
           0.1  0.2  0.6];

% Cholesky decomposition for shock simulation
L = chol(Sigma_u, 'lower');

% Generate VAR data
data = zeros(T, K);
data(1:nlags, :) = randn(nlags, K);  % Initial values

for t = (nlags+1):T
    shock = L * randn(K, 1);
    data(t, :) = nu' + data(t-1, :) * A1' + data(t-2, :) * A2' + shock';
end

disp('========================================');
disp('Data simulation complete');
disp(['Sample size: ' num2str(T)]);
disp(['Number of variables: ' num2str(K)]);
disp('========================================');
fprintf('\n');

%% ========================================================================
%% EXAMPLE 1: Basic VAR with constant
%% ========================================================================
disp('========================================');
disp('EXAMPLE 1: Basic VAR(2) with constant');
disp('========================================');

EQ1 = reducedformVAR(data, 2, true);

disp('Estimated constants (vs true values):');
disp('           Estimated    True');
for i = 1:K
    fprintf('Var %d:     %8.4f   %8.4f\n', i, EQ1.constants(i), nu(i));
end
fprintf('\n');

disp('Estimated A1 matrix (first lag):');
disp(EQ1.coefficients(:, 1:K));
disp('True A1 matrix:');
disp(A1);
fprintf('\n');

disp('Estimated A2 matrix (second lag):');
disp(EQ1.coefficients(:, K+1:2*K));
disp('True A2 matrix:');
disp(A2);
fprintf('\n');

disp('Model selection criteria:');
fprintf('AIC: %.4f\n', EQ1.AIC);
fprintf('BIC: %.4f\n', EQ1.BIC);
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 2: VAR with restrictions on variable interactions
%% ========================================================================
disp('========================================');
disp('EXAMPLE 2: Restricted VAR');
disp('Restriction: Variable 3 does not affect variables 1 and 2');
disp('========================================');

% Restriction matrix: zeros mean no effect
lr = [1 1 0;   % Var 1 affected by vars 1,2 only (not by 3)
      1 1 0;   % Var 2 affected by vars 1,2 only (not by 3)
      1 1 1];  % Var 3 affected by all variables

EQ2 = reducedformVAR(data, 2, true, lr);

disp('Unrestricted: coefficients on variable 3 in equations 1-2:');
disp('         Lag 1    Lag 2');
for i = 1:2
    fprintf('Eq %d:   %7.4f  %7.4f\n', i, EQ1.coefficients(i, 3), ...
            EQ1.coefficients(i, 6));
end
fprintf('\n');

disp('Restricted: coefficients on variable 3 in equations 1-2 (should be zero):');
disp('         Lag 1    Lag 2');
for i = 1:2
    fprintf('Eq %d:   %7.4f  %7.4f\n', i, EQ2.coefficients(i, 3), ...
            EQ2.coefficients(i, 6));
end
fprintf('\n');

disp('Model comparison:');
fprintf('Unrestricted BIC: %.4f\n', EQ1.BIC);
fprintf('Restricted BIC:   %.4f\n', EQ2.BIC);
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 3: VAR with forecasting
%% ========================================================================
disp('========================================');
disp('EXAMPLE 3: VAR with out-of-sample forecast');
disp('========================================');

EQ3 = reducedformVAR(data, 2, true, [], true);

forecast_horizon = size(EQ3.forecast.cent, 2) - T + 1;
fprintf('Forecast horizon: %d periods\n', forecast_horizon);
fprintf('\n');
close all

% Plot forecasts for all variables
figure('Position', [100 100 1200 400]);
for i = 1:K
    subplot(1, K, i);
    hold on;
    
    % In-sample fit
    plot(1:T-1, EQ3.forecast.cent(i, 1:T-1), 'b-', 'LineWidth', 1.5);
    
    % Out-of-sample forecast
    plot(T:size(EQ3.forecast.cent,2), EQ3.forecast.cent(i, T:end), ...
         'r-', 'LineWidth', 1.5);
    
    % Confidence bands
    plot(T:size(EQ3.forecast.cent,2), EQ3.forecast.conf95(i, T:end), ...
         'r--', 'LineWidth', 1);
    plot(T:size(EQ3.forecast.cent,2), EQ3.forecast.conf05(i, T:end), ...
         'r--', 'LineWidth', 1);
    
    % Vertical line at forecast start
    xline(T, 'k--', 'LineWidth', 1.5);
    
    xlabel('Time');
    ylabel(['Variable ' num2str(i)]);
    title(['Variable ' num2str(i) ': Forecast']);
    legend('In-sample', 'Forecast', '90% CI', '', 'Location', 'best');
    grid on;
    hold off;
end
sgtitle('Out-of-sample forecasts with 90% confidence intervals');

disp('Forecast plot created (Figure 1)');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 4: VAR with seasonal dummies
%% ========================================================================
disp('========================================');
disp('EXAMPLE 4: VAR with quarterly seasonal dummies');
disp('========================================');

% Simulate quarterly data with seasonal pattern
T_q = 120;  % 30 years of quarterly data
data_seasonal = zeros(T_q, K);
data_seasonal(1:nlags, :) = randn(nlags, K);

% Create seasonal effects (Q1, Q2, Q3, Q4)
seasonal_pattern = [0.5; -0.3; 0.2; -0.4];  % Different effect each quarter

for t = (nlags+1):T_q
    shock = L * randn(K, 1);
    quarter = mod(t-1, 4) + 1;  % Which quarter (1-4)
    seasonal_effect = seasonal_pattern(quarter) * [1; 0.5; 0.3]';  % Different effect per variable
    
    data_seasonal(t, :) = nu' + data_seasonal(t-1, :) * A1' + ...
                          data_seasonal(t-2, :) * A2' + seasonal_effect + shock';
end

EQ4 = reducedformVAR(data_seasonal, 2, true, [], false, 'quarter');

disp('Estimated seasonal dummy coefficients:');
disp('         Q1        Q2        Q3');
disp(EQ4.dummies);
fprintf('Last period in sample corresponds to quarter: %d\n', EQ4.lastperiod);
fprintf('\n');

disp('T-statistics for seasonal dummies:');
disp(EQ4.tstats_dumm);
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 5: VAR with exogenous variables
%% ========================================================================
disp('========================================');
disp('EXAMPLE 5: VAR with exogenous dummy variable');
disp('Crisis dummy for observations 80-90');
disp('========================================');

% Create crisis dummy
exog = zeros(T, 1);
exog(80:90) = 1;

% Add crisis effect to data
data_crisis = data;
crisis_effect = [-2; -1.5; -1];  % Negative shock during crisis
for t = 80:90
    data_crisis(t, :) = data_crisis(t, :) + crisis_effect';
end

% Estimate without exogenous variable
EQ5a = reducedformVAR(data_crisis, 2, true);

% Estimate with exogenous variable
EQ5b = reducedformVAR(data_crisis, 2, true, [], false, [], exog);

disp('Exogenous variable coefficients (crisis dummy):');
disp('           Coef      T-stat');
for i = 1:K
    fprintf('Var %d:   %7.3f   %7.3f\n', i, EQ5b.exo(i), EQ5b.tstats_exo(i));
end
fprintf('\n');

disp('Residual variance comparison (trace of Sigma):');
fprintf('Without crisis dummy: %.4f\n', trace(EQ5a.sigma));
fprintf('With crisis dummy:    %.4f\n', trace(EQ5b.sigma));
fprintf('Improvement:          %.2f%%\n', ...
        100*(trace(EQ5a.sigma) - trace(EQ5b.sigma))/trace(EQ5a.sigma));
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 6: Combining multiple features
%% ========================================================================
disp('========================================');
disp('EXAMPLE 6: Combined features');
disp('VAR with constant, restrictions, and forecast');
disp('========================================');

lr_combined = [1 1 0;
               1 1 0;
               1 1 1];

EQ6 = reducedformVAR(data, 2, true, lr_combined, true);
close all

fprintf('Model estimated with:\n');
fprintf('  - Constant term: YES\n');
fprintf('  - Restrictions: Variable 3 does not affect variables 1-2\n');
fprintf('  - Forecast: YES (%d periods ahead)\n', ...
        size(EQ6.forecast.cent, 2) - T + 1);
fprintf('\n');

disp('Forecast summary for Variable 1:');
fprintf('Last observed value:    %.4f\n', data(end, 1));
fprintf('1-period ahead forecast: %.4f [%.4f, %.4f]\n', ...
        EQ6.forecast.cent(1, T), EQ6.forecast.conf05(1, T), ...
        EQ6.forecast.conf95(1, T));
fprintf('\n\n');

disp('========================================');
disp('All examples completed successfully!');
disp('========================================');