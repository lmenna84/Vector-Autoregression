%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% EXAMPLES FOR nolucas
%% Author: Lorenzo Menna
%% This script demonstrates the nolucas function for counterfactual
%% historical decomposition with channel restrictions
%% WARNING: Results are NOT robust to Lucas critique when channels are blocked
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all; clc;

%% ========================================================================
%% DATA SIMULATION
%% ========================================================================
% We simulate a 4-variable VAR to demonstrate channel-blocking analysis
% Variables: Output, Inflation, Interest Rate, Exchange Rate

rng(12345);  % Set seed for reproducibility

% Parameters
K = 4;        % Number of variables
T = 250;      % Sample size
nlags = 2;    % Number of lags

% True VAR coefficient matrices (stable system, spectral radius < 1)
A1 = [0.5  0.1  -0.1  0.0;
      0.1  0.5   0.0  0.0;
      0.1  0.2   0.4  0.0;
      0.1  0.0   0.1  0.4];

A2 = [0.1  0.0  0.0  0.0;
      0.0  0.1  0.0  0.0;
      0.0  0.0  0.1  0.0;
      0.0  0.0  0.0  0.1];

nu = [0.5; 1.0; 0.5; 1.0];  % Constants

% True structural impact matrix (Cholesky for identification)
B_true = [1.0  0.0  0.0  0.0;
          0.3  0.8  0.0  0.0;
          0.2  0.4  0.6  0.0;
          0.1  0.05  0.02  0.7];

% Generate structural shocks (orthogonal)
epsilon = randn(K, T);

% Generate VAR data
data = zeros(T, K);
data(1:nlags,:) = randn(nlags, K);  % Initial values

for t = (nlags+1):T
    structural_shock = B_true * epsilon(:,t);
    data(t,:) = nu' + data(t-1,:)*A1' + data(t-2,:)*A2' + structural_shock';
end

var_names = {'Output', 'Inflation', 'InterestRate', 'ExchangeRate'};
shock_names = {'SupplyShock', 'DemandShock', 'MonetaryShock', 'ExchangeShock'};

disp('========================================');
disp('Data simulation complete');
disp(['Sample size: ' num2str(T) ' observations']);
disp(['Variables: ' strjoin(var_names, ', ')]);
disp('Identification: Recursive (Cholesky)');
disp('========================================');
fprintf('\n');

%% ========================================================================
%% ESTIMATE SVAR USING srSVAR
%% ========================================================================
disp('========================================');
disp('Estimating baseline SVAR (Cholesky identification)');
disp('========================================');

EQ = srSVAR(data, nlags, var_names, 40, 1000, true);
close all;

disp('SVAR estimation complete');
fprintf('Impact matrix P dimensions: %dx%d\n', size(EQ.P,1), size(EQ.P,2));
fprintf('Structural shocks dimensions: %dx%d\n', size(EQ.struc,1), size(EQ.struc,2));
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 1: Standard historical decomposition (no channels blocked)
%% ========================================================================
disp('========================================');
disp('EXAMPLE 1: Standard historical decomposition');
disp('Contribution of monetary shock with ALL transmission channels active');
disp('========================================');

% Extract VAR components from srSVAR output
Q = reducedformVAR(data, nlags, true);  % Re-estimate to get coefficients

withvar = ones(1,K);  % Allow all transmissions
withconst = 1;        % Full deterministic part
shock_idx = 3;        % Monetary shock

EQ1 = nolucas(data, Q.constants, Q.coefficients, EQ.P, EQ.struc, ...
              shock_idx, withvar, withconst);

% Plot decomposition
figure('Position', [100 100 1200 800]);
for i = 1:K
    subplot(K, 1, i);
    hold on;
    plot(1:T, data(:,i), 'k-', 'LineWidth', 1.5, 'DisplayName', 'Actual Data');
    plot(1:T, EQ1.contribution(i,:), 'b--', 'LineWidth', 1.5, ...
         'DisplayName', 'Contribution (Shock + Determ)');
    plot(1:T, EQ1.onlyshock(i,:), 'r:', 'LineWidth', 1.5, ...
         'DisplayName', 'Only Shock');
    plot(1:T, EQ1.deterministic(i,:), 'g-.', 'LineWidth', 1.5, ...
         'DisplayName', 'Only Deterministic');
    plot(1:T, zeros(1,T), 'k--', 'LineWidth', 0.5);
    title([var_names{i} ' - Monetary Shock Contribution (All Channels)']);
    xlabel('Time');
    ylabel(var_names{i});
    legend('Location', 'best');
    grid on;
end
sgtitle('Standard Historical Decomposition: Monetary Shock');
fprintf('Standard decomposition plot created (Figure 1)\n');

% -----------------------------------------------------------------
% IDENTITY CHECK: sum of all shock contributions + deterministic = data
% Run nolucas for each shock (all channels open), then verify the identity
% -----------------------------------------------------------------
disp('--- Identity check: sum of shocks + deterministic = data ---');
sum_shocks = zeros(K, T);
for s = 1:K
    EQ_s = nolucas(data, Q.constants, Q.coefficients, EQ.P, EQ.struc, ...
                   s, ones(1,K), 1);
    sum_shocks = sum_shocks + EQ_s.onlyshock;
end
% deterministic is the same for every shock; reuse EQ_s.deterministic
reconstructed = sum_shocks + EQ_s.deterministic;

% Maximum absolute reconstruction error (should be machine-precision)
max_err = max(max(abs(reconstructed - data')));
fprintf('Max absolute error |sum(shocks) + deterministic - data|: %.2e\n', max_err);
if max_err < 1e-8
    disp('Identity verified: reconstruction matches data to machine precision.');
else
    disp('WARNING: reconstruction error larger than expected.');
end
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 2: Block interest rate channel
%% ========================================================================
disp('========================================');
disp('EXAMPLE 2: Monetary shock WITHOUT interest rate transmission');
disp('Counterfactual: What if monetary policy could not affect variables');
disp('through the interest rate channel?');
disp('========================================');

withvar = [1 1 0 1];  % Block transmission through InterestRate (variable 3)
withconst = 0;        % Also block interest rate in deterministic part

EQ2 = nolucas(data, Q.constants, Q.coefficients, EQ.P, EQ.struc, ...
              shock_idx, withvar, withconst);

% Compare with standard decomposition
figure('Position', [100 100 1200 600]);
var_to_plot = [1, 2];  % Output and Inflation
titles = {'Output', 'Inflation'};

for idx = 1:2
    i = var_to_plot(idx);
    subplot(2, 2, idx);
    hold on;
    plot(1:T, EQ1.onlyshock(i,:), 'b-', 'LineWidth', 1.5, ...
         'DisplayName', 'All Channels');
    plot(1:T, EQ2.onlyshock(i,:), 'r--', 'LineWidth', 1.5, ...
         'DisplayName', 'Interest Rate Blocked');
    plot(1:T, zeros(1,T), 'k--', 'LineWidth', 0.5);
    title([titles{idx} ' Response - Monetary Shock']);
    xlabel('Time');
    ylabel(titles{idx});
    legend('Location', 'best');
    grid on;
    
    % Contribution at specific date
    subplot(2, 2, idx+2);
    dates_plot = 100:150;
    hold on;
    plot(dates_plot, EQ1.onlyshock(i,dates_plot), 'b-', 'LineWidth', 2, ...
         'DisplayName', 'All Channels');
    plot(dates_plot, EQ2.onlyshock(i,dates_plot), 'r--', 'LineWidth', 2, ...
         'DisplayName', 'Interest Rate Blocked');
    plot(dates_plot, zeros(size(dates_plot)), 'k--', 'LineWidth', 0.5);
    title([titles{idx} ' - Detail (t=100-150)']);
    xlabel('Time');
    ylabel(titles{idx});
    legend('Location', 'best');
    grid on;
end
sgtitle('Blocking Interest Rate Channel');
fprintf('Channel-blocking comparison plot created (Figure 2)\n');
fprintf('\n');

% Quantify the role of interest rate channel
role_output = mean(abs(EQ1.onlyshock(1,50:end) - EQ2.onlyshock(1,50:end)));
role_inflation = mean(abs(EQ1.onlyshock(2,50:end) - EQ2.onlyshock(2,50:end)));
fprintf('Average absolute difference (ignoring first 50 periods):\n');
fprintf('  Output:    %.4f (role of interest rate channel)\n', role_output);
fprintf('  Inflation: %.4f (role of interest rate channel)\n', role_inflation);
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 3: Block exchange rate channel
%% ========================================================================
disp('========================================');
disp('EXAMPLE 3: Monetary shock WITHOUT exchange rate transmission');
disp('Counterfactual: Closed economy (no exchange rate effects)');
disp('========================================');

withvar = [1 1 1 0];  % Block transmission through ExchangeRate (variable 4)
withconst = 1;        % Keep exchange rate in deterministic part

EQ3 = nolucas(data, Q.constants, Q.coefficients, EQ.P, EQ.struc, ...
              shock_idx, withvar, withconst);

fprintf('Average contribution to Output (periods 50-250):\n');
fprintf('  All channels:             %.4f\n', mean(EQ1.onlyshock(1,50:end)));
fprintf('  Exchange rate blocked:    %.4f\n', mean(EQ3.onlyshock(1,50:end)));
fprintf('  Interest rate blocked:    %.4f\n', mean(EQ2.onlyshock(1,50:end)));
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 4: Block multiple channels simultaneously
%% ========================================================================
disp('========================================');
disp('EXAMPLE 4: Monetary shock with ONLY direct effects');
disp('Block BOTH interest rate AND exchange rate channels');
disp('========================================');

withvar = [1 1 0 0];  % Block both InterestRate and ExchangeRate
withconst = 0;

EQ4 = nolucas(data, Q.constants, Q.coefficients, EQ.P, EQ.struc, ...
              shock_idx, withvar, withconst);

% Create decomposition of total effect
figure('Position', [100 100 1200 400]);
var_idx = 1;  % Focus on Output

subplot(1, 2, 1);
hold on;
plot(100:150, EQ1.onlyshock(var_idx,100:150), 'k-', 'LineWidth', 2.5, ...
     'DisplayName', 'Total Effect');
plot(100:150, EQ4.onlyshock(var_idx,100:150), 'b--', 'LineWidth', 2, ...
     'DisplayName', 'Direct Effect Only');
plot(100:150, EQ1.onlyshock(var_idx,100:150) - EQ4.onlyshock(var_idx,100:150), ...
     'r:', 'LineWidth', 2, 'DisplayName', 'Indirect Effects');
plot(100:150, zeros(1,51), 'k--', 'LineWidth', 0.5);
title('Output: Decomposition of Monetary Shock Effects');
xlabel('Time');
ylabel('Output');
legend('Location', 'best');
grid on;

subplot(1, 2, 2);
indirect = EQ1.onlyshock(var_idx,:) - EQ4.onlyshock(var_idx,:);
direct = EQ4.onlyshock(var_idx,:);
total = EQ1.onlyshock(var_idx,:);
bar_data = [mean(abs(direct(50:end))), mean(abs(indirect(50:end)))];
bar(bar_data);
set(gca, 'XTickLabel', {'Direct', 'Indirect (via rate + FX)'});
ylabel('Average Absolute Contribution');
title('Direct vs Indirect Effects (Output)');
grid on;

sgtitle('Channel Decomposition: Monetary Shock');
fprintf('Channel decomposition plot created (Figure 3)\n');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 5: Different shocks, different channels
%% ========================================================================
disp('========================================');
disp('EXAMPLE 5: Analyzing different shocks with channel restrictions');
disp('========================================');

% Supply shock (shock 1) without exchange rate channel
EQ5a = nolucas(data, Q.constants, Q.coefficients, EQ.P, EQ.struc, ...
               1, [1 1 1 0], 1);

% Demand shock (shock 2) without interest rate channel  
EQ5b = nolucas(data, Q.constants, Q.coefficients, EQ.P, EQ.struc, ...
               2, [1 1 0 1], 1);

fprintf('Inflation contribution at t=100:\n');
fprintf('  Supply shock (ExRate blocked):  %.4f\n', EQ5a.onlyshock(2,100));
fprintf('  Demand shock (IntRate blocked): %.4f\n', EQ5b.onlyshock(2,100));
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 6: withconst parameter
%% ========================================================================
disp('========================================');
disp('EXAMPLE 6: Understanding the withconst parameter');
disp('========================================');

withvar = [1 1 0 1];  % Block interest rate channel
shock_idx = 3;

% withconst=1: Interest rate affects deterministic part
EQ6a = nolucas(data, Q.constants, Q.coefficients, EQ.P, EQ.struc, ...
               shock_idx, withvar, 1);

% withconst=0: Interest rate also blocked in deterministic
EQ6b = nolucas(data, Q.constants, Q.coefficients, EQ.P, EQ.struc, ...
               shock_idx, withvar, 0);

fprintf('Deterministic component for Output at t=100:\n');
fprintf('  withconst=1 (full A matrix):       %.4f\n', EQ6a.deterministic(1,100));
fprintf('  withconst=0 (restricted A matrix): %.4f\n', EQ6b.deterministic(1,100));
fprintf('  Difference:                        %.4f\n', ...
        EQ6a.deterministic(1,100) - EQ6b.deterministic(1,100));
fprintf('\n');
disp('withconst=1: Excluded variables still affect autoregressive dynamics');
disp('withconst=0: Excluded variables fully shut down (more restrictive)');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 7: Forecasting with channel restrictions
%% ========================================================================
disp('========================================');
disp('EXAMPLE 7: Counterfactual forecasting');
disp('Forecast what would happen if interest rate channel were blocked');
disp('========================================');

forecast_horizon = 20;
withvar = [1 1 0 1];
shock_idx = 3;

EQ7 = nolucas(data, Q.constants, Q.coefficients, EQ.P, EQ.struc, ...
              shock_idx, withvar, 0, [], [], [], [], forecast_horizon);

% Plot historical + forecast
figure('Position', [100 100 1200 600]);
T_plot_start = T-50;
for i = 1:2  % Output and Inflation
    subplot(2, 1, i);
    hold on;
    % Historical
    plot(T_plot_start:T, EQ7.onlyshock(i,T_plot_start:T), 'b-', 'LineWidth', 2, ...
         'DisplayName', 'Historical');
    % Forecast
    plot(T+1:T+forecast_horizon, EQ7.forecast_onlyshock(i,:), 'r--', ...
         'LineWidth', 2, 'DisplayName', 'Forecast');
    plot([T T], ylim, 'k--', 'LineWidth', 1.5);
    plot(T_plot_start:T+forecast_horizon, zeros(1, 51+forecast_horizon), ...
         'k:', 'LineWidth', 0.5);
    title([var_names{i} ' - Monetary Shock (Interest Rate Channel Blocked)']);
    xlabel('Time');
    ylabel(var_names{i});
    legend('Location', 'best');
    grid on;
end
sgtitle('Counterfactual Forecast: Monetary Shock');
fprintf('Forecast plot created (Figure 4)\n');
fprintf('\n');
fprintf('Forecast horizon: %d periods\n', forecast_horizon);
fprintf('Last historical contribution (Output): %.4f\n', EQ7.onlyshock(1,end));
fprintf('Final forecast (Output):               %.4f\n', ...
        EQ7.forecast_onlyshock(1,end));
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 8: With seasonal dummies
%% ========================================================================
disp('========================================');
disp('EXAMPLE 8: Historical decomposition with quarterly dummies');
disp('========================================');

% Re-estimate VAR with seasonal dummies
EQ_dum = srSVAR(data, nlags, var_names, 40, 1000, true, [], 'quarter');
close all;
Q_dum = reducedformVAR(data, nlags, true, [], [], 'quarter');

withvar = ones(1,K);
shock_idx = 2;  % Demand shock

EQ8 = nolucas(data, Q_dum.constants, Q_dum.coefficients, EQ_dum.P, ...
              EQ_dum.struc, shock_idx, withvar, 1, 'quarter', Q_dum.dummies);

fprintf('Seasonal dummies included in deterministic component\n');
fprintf('Contribution properly adjusted for seasonal patterns\n');
fprintf('Mean contribution to Inflation (periods 50-250): %.4f\n', ...
        mean(EQ8.onlyshock(2,50:end)));
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 9: With exogenous variables
%% ========================================================================
disp('========================================');
disp('EXAMPLE 9: Historical decomposition with exogenous oil shock');
disp('========================================');

% Create oil shock dummy
exog = zeros(T, 1);
exog(120:130) = 1;  % Oil crisis period

% Add oil effect to data
data_oil = data;
oil_effect = [0.5; 1.0; 0.3; -2.0];
for t = 120:130
    data_oil(t,:) = data_oil(t,:) + oil_effect';
end

% Estimate VAR with exogenous variable
EQ_oil = srSVAR(data_oil, nlags, var_names, 40, 1000, true, [], [], exog);
close all;
Q_oil = reducedformVAR(data_oil, nlags, true, [], [], [], exog);

withvar = ones(1,K);
shock_idx = 2;

EQ9 = nolucas(data_oil, Q_oil.constants, Q_oil.coefficients, EQ_oil.P, ...
              EQ_oil.struc, shock_idx, withvar, 1, [], [], exog, Q_oil.exo);

fprintf('Oil shock controlled via exogenous dummy\n');
fprintf('Decomposition properly accounts for oil crisis (t=120-130)\n');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 10: Lucas critique WARNING
%% ========================================================================
disp('========================================');
disp('EXAMPLE 10: Lucas critique WARNING');
disp('========================================');

disp('CRITICAL CAVEAT: Channel-blocking analysis is NOT robust to Lucas critique!');
fprintf('\n');
disp('When you block a transmission channel, you assume:');
disp('  1. The VAR coefficients remain unchanged');
disp('  2. Other channels function exactly as before');
disp('  3. Agents do not respond to the policy change');
fprintf('\n');
disp('In reality, if you actually shut down the interest rate channel:');
disp('  - Central bank would change its policy rule');
disp('  - Private agents would adjust expectations');
disp('  - Other transmission channels would change');
disp('  - VAR coefficients would be different');
fprintf('\n');
disp('APPROPRIATE USES of channel-blocking:');
disp('  ✓ Diagnostic tool: "How important is channel X historically?"');
disp('  ✓ Decomposition: "What share of the effect came through channel X?"');
disp('  ✓ Comparison: "Which channel matters more for Output vs Inflation?"');
fprintf('\n');
disp('INAPPROPRIATE USES:');
disp('  ✗ Policy recommendation: "We should block channel X"');
disp('  ✗ Counterfactual policy: "If we had blocked X, Y would have happened"');
disp('  ✗ Structural interpretation: "Channel X is unnecessary"');
fprintf('\n');
disp('This is a DIAGNOSTIC tool, not a structural model for policy evaluation!');
fprintf('\n\n');

%% ========================================================================
disp('========================================');
disp('All examples completed successfully!');
disp('========================================');
disp('Key nolucas features demonstrated:');
disp('- Standard historical decomposition (no channels blocked)');
disp('- Blocking single transmission channels (interest rate, exchange rate)');
disp('- Blocking multiple channels simultaneously');
disp('- Direct vs indirect effect decomposition');
disp('- Different shocks with different channel restrictions');
disp('- Understanding withconst parameter');
disp('- Counterfactual forecasting');
disp('- Seasonal dummies support');
disp('- Exogenous variables support');
disp('- Lucas critique warning');
disp(' ');
disp('Key concepts:');
disp('- onlyshock: Pure contribution of shock (with restricted channels)');
disp('- deterministic: Constants + dummies + exog + autoregressive dynamics');
disp('- contribution: Total = onlyshock + deterministic');
disp('- withvar: 1=allow transmission, 0=block transmission');
disp('- withconst: 1=excluded vars affect deterministic, 0=fully blocked');
disp('- NOT robust to Lucas critique when channels are blocked!');