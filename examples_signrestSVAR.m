%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% EXAMPLES FOR signrestSVAR
%% Author: Lorenzo Menna
%% This script demonstrates all features of the signrestSVAR function
%% (Sign restrictions identification - Uhlig 2005, Rubio-Ramirez et al. 2010)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all; clc;

%% ========================================================================
%% DATA SIMULATION
%% ========================================================================
% We simulate a 3-variable VAR(2) system for sign restrictions identification

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

% True structural impact matrix
B_true = [1.0  0.0  0.0;
          0.5  1.2  0.0;
          0.3  0.4  0.8];

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
disp('Sign restrictions focus on identification uncertainty');
disp('========================================');
fprintf('\n');

%% ========================================================================
%% EXAMPLE 1: Basic sign restrictions SVAR (impact only)
%% ========================================================================
disp('========================================');
disp('EXAMPLE 1: Basic sign restrictions - monetary policy shock');
disp('Impact restrictions only (periods_sign = 1)');
disp('========================================');

var_names = {'Output', 'Inflation', 'InterestRate'};
shock_names = {'Supply', 'Demand', 'MonetaryPolicy'};

% Sign restriction matrix: element (i,j) = sign of response of variable i to shock j
% Monetary policy shock (3rd column):
%   - Output decreases (negative)
%   - Inflation decreases (negative)  
%   - Interest rate increases (positive)
sign_matrix = [0   0  -1;    % Output: no restriction on Supply/Demand, negative for MP
               0   0  -1;    % Inflation: negative for MP
               0   0   1];   % Interest rate: positive for MP

periods_sign = 1;  % Only impact restrictions

EQ1 = signrestSVAR(data, nlags, sign_matrix, periods_sign, var_names, ...
                   shock_names, 40, 1000, true);
close all;

disp('Sign restrictions imposed:');
disp('  Monetary Policy shock (column 3):');
disp('    - Output ↓ (negative impact)');
disp('    - Inflation ↓ (negative impact)');
disp('    - Interest Rate ↑ (positive impact)');
fprintf('\n');
disp('Note: IRFs reported are MEDIAN across accepted models');
disp('This accounts for identification uncertainty');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 2: Multi-period sign restrictions
%% ========================================================================
disp('========================================');
disp('EXAMPLE 2: Multi-period sign restrictions');
disp('Restrictions enforced for first 4 periods');
disp('========================================');

% Same restrictions but enforced for 4 periods
periods_sign = 4;

EQ2 = signrestSVAR(data, nlags, sign_matrix, periods_sign, var_names, ...
                   shock_names, 40, 1000, true);
close all;

disp('Sign restrictions enforced at horizons 0, 1, 2, 3');
disp('This produces tighter identification');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 3: Multiple shocks identified simultaneously
%% ========================================================================
disp('========================================');
disp('EXAMPLE 3: Identifying multiple shocks with sign restrictions');
disp('Supply, Demand, and Monetary shocks all identified');
disp('========================================');

% Full sign restriction matrix identifying all three shocks:
% Supply shock: Output ↑, Inflation ↓
% Demand shock: Output ↑, Inflation ↑
% Monetary shock: Output ↓, Inflation ↓, Interest rate ↑
sign_full = [ 1   1  -1;    % Output
             -1   1  -1;    % Inflation
              0   0   1];   % Interest rate

EQ3 = signrestSVAR(data, nlags, sign_full, 1, var_names, shock_names, ...
                   40, 1000, true);
close all;

disp('All three shocks identified via sign restrictions:');
disp('  Supply shock:    Output ↑, Inflation ↓');
disp('  Demand shock:    Output ↑, Inflation ↑');
disp('  Monetary shock:  Output ↓, Inflation ↓, Rate ↑');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 4: With variance decomposition
%% ========================================================================
disp('========================================');
disp('EXAMPLE 4: Sign restrictions with variance decomposition');
disp('========================================');

EQ4 = signrestSVAR(data, nlags, sign_full, 1, var_names, shock_names, ...
                   40, 1000, true, [], [], [], [], 5, 1);  % vardec=1
close all;

disp('Variance decomposition of Output (median):');
disp('Horizon    Supply     Demand     Monetary');
for h = [1, 4, 12, 20, 40]
    fprintf('%3d        %.2f%%     %.2f%%     %.2f%%\n', h, ...
            EQ4.vardecshock_Supply(1,h), ...
            EQ4.vardecshock_Demand(1,h), ...
            EQ4.vardecshock_MonetaryPolicy(1,h));
end
fprintf('\n');

% Plot variance decomposition
figure('Position', [100 100 1200 400]);
subplot(1, 3, 1);
plot(1:40, EQ4.vardecshock_Supply(1,:), 'k-', 'LineWidth', 2);
hold on;
plot(1:40, EQ4.vardecshock_Demand(1,:), 'k--', 'LineWidth', 2);
plot(1:40, EQ4.vardecshock_MonetaryPolicy(1,:), 'k:', 'LineWidth', 2);
title('Output Variance Decomposition');
xlabel('Horizon');
ylabel('Percentage');
legend('Supply', 'Demand', 'Monetary', 'Location', 'best');
ylim([0 105]);
grid on;

subplot(1, 3, 2);
plot(1:40, EQ4.vardecshock_Supply(2,:), 'k-', 'LineWidth', 2);
hold on;
plot(1:40, EQ4.vardecshock_Demand(2,:), 'k--', 'LineWidth', 2);
plot(1:40, EQ4.vardecshock_MonetaryPolicy(2,:), 'k:', 'LineWidth', 2);
title('Inflation Variance Decomposition');
xlabel('Horizon');
ylabel('Percentage');
legend('Supply', 'Demand', 'Monetary', 'Location', 'best');
ylim([0 105]);
grid on;

subplot(1, 3, 3);
plot(1:40, EQ4.vardecshock_Supply(3,:), 'k-', 'LineWidth', 2);
hold on;
plot(1:40, EQ4.vardecshock_Demand(3,:), 'k--', 'LineWidth', 2);
plot(1:40, EQ4.vardecshock_MonetaryPolicy(3,:), 'k:', 'LineWidth', 2);
title('Interest Rate Variance Decomposition');
xlabel('Horizon');
ylabel('Percentage');
legend('Supply', 'Demand', 'Monetary', 'Location', 'best');
ylim([0 105]);
grid on;

sgtitle('Variance Decomposition (Sign Restrictions)');
fprintf('Variance decomposition plot created (Figure 1)\n');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 5: Custom shock sizes
%% ========================================================================
disp('========================================');
disp('EXAMPLE 5: Sign restrictions with custom shock sizes');
disp('========================================');

% 2 unit shocks for all variables
shock_size = 2 * ones(K, 1);

EQ5 = signrestSVAR(data, nlags, sign_matrix, 1, var_names, shock_names, ...
                   40, 1000, true, [], [], [], shock_size);
close all;

disp('Custom shock sizes applied');
disp('All shocks scaled to 2 units');
fprintf('\n');

% Compare with 1-unit shocks
disp('Comparison at horizon 10:');
fprintf('1-unit shock:  %.4f\n', EQ1.Output_MonetaryPolicy(2,10));
fprintf('2-unit shock:  %.4f\n', EQ5.Output_MonetaryPolicy(2,10));
fprintf('Ratio:         %.4f (expected: ~2.0)\n', ...
        EQ5.Output_MonetaryPolicy(2,10) / EQ1.Output_MonetaryPolicy(2,10));
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 6: Different confidence levels
%% ========================================================================
disp('========================================');
disp('EXAMPLE 6: Sign restrictions with different confidence intervals');
disp('========================================');

EQ6a = signrestSVAR(data, nlags, sign_matrix, 1, var_names, shock_names, ...
                    40, 1000, true, [], [], [], [], 5);    % 90% CI
EQ6b = signrestSVAR(data, nlags, sign_matrix, 1, var_names, shock_names, ...
                    40, 1000, true, [], [], [], [], 16);   % 68% CI
close all;

disp('Output response to Monetary Policy shock at horizon 10:');
fprintf('90%% CI: [%.4f, %.4f]  Width: %.4f\n', ...
        EQ6a.Output_MonetaryPolicy(3,10), EQ6a.Output_MonetaryPolicy(1,10), ...
        EQ6a.Output_MonetaryPolicy(1,10) - EQ6a.Output_MonetaryPolicy(3,10));
fprintf('68%% CI: [%.4f, %.4f]  Width: %.4f\n', ...
        EQ6b.Output_MonetaryPolicy(3,10), EQ6b.Output_MonetaryPolicy(1,10), ...
        EQ6b.Output_MonetaryPolicy(1,10) - EQ6b.Output_MonetaryPolicy(3,10));
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 7: With seasonal dummies
%% ========================================================================
disp('========================================');
disp('EXAMPLE 7: Sign restrictions with quarterly seasonal dummies');
disp('========================================');

EQ7 = signrestSVAR(data, nlags, sign_matrix, 1, var_names, shock_names, ...
                   40, 1000, true, [], 'quarter');
close all;

disp('Seasonal patterns controlled for in estimation');
disp('Sign restrictions applied to deseasonalized shocks');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 8: With exogenous variables
%% ========================================================================
disp('========================================');
disp('EXAMPLE 8: Sign restrictions with exogenous oil price shock');
disp('========================================');

% Create oil price shock dummy
exog = zeros(T, 1);
exog(80:85) = 1;  % Oil shock period

% Add oil shock effect to data
data_oil = data;
oil_effect = B_true * [0; 0.5; 1];  % Affects inflation and interest rate
for t = 80:85
    data_oil(t, :) = data_oil(t, :) + oil_effect';
end

EQ8 = signrestSVAR(data_oil, nlags, sign_matrix, 1, var_names, shock_names, ...
                   40, 1000, true, [], [], exog);
close all;

disp('Oil price shock controlled for via exogenous dummy');
disp('Sign restrictions applied to oil-adjusted data');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 9: With restricted reduced-form VAR
%% ========================================================================
disp('========================================');
disp('EXAMPLE 9: Sign restrictions with restricted reduced-form VAR');
disp('========================================');

lr = [1 1 0;   % Output affected by output, inflation only
      1 1 0;   % Inflation affected by output, inflation only
      1 1 1];  % Interest rate affected by all variables

EQ9 = signrestSVAR(data, nlags, sign_matrix, 1, var_names, shock_names, ...
                   40, 1000, true, lr);
close all;

disp('Reduced-form restrictions on lags imposed via lr matrix');
disp('Sign restrictions applied to restricted VAR');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 10: Understanding identification uncertainty
%% ========================================================================
disp('========================================');
disp('EXAMPLE 10: Examining identification uncertainty');
disp('Using EQ.irfsim to see distribution of accepted models');
disp('========================================');

% Run with fewer repetitions for speed
EQ10 = signrestSVAR(data, nlags, sign_matrix, 1, var_names, shock_names, ...
                    40, 500, true);
close all;

% Extract all accepted models for Monetary Policy shock effect on Output at h=10
% irfsim dimensions: [variable, horizon, repetition, shock]
irf_draws = squeeze(EQ10.irfsim(1, 10, :, 3));  % Output, h=10, all reps, MP shock

disp('Distribution of IRFs across accepted models:');
fprintf('  Median: %.4f\n', median(irf_draws));
fprintf('  Mean:   %.4f\n', mean(irf_draws));
fprintf('  Std:    %.4f\n', std(irf_draws));
fprintf('  Min:    %.4f\n', min(irf_draws));
fprintf('  Max:    %.4f\n', max(irf_draws));
fprintf('\n');

% Plot distribution
figure('Position', [100 100 1000 400]);
subplot(1, 2, 1);
histogram(irf_draws, 30, 'Normalization', 'pdf', 'FaceColor', [0.7 0.7 0.7]);
hold on;
xline(median(irf_draws), 'r-', 'LineWidth', 2, 'Label', 'Median (reported)');
xline(prctile(irf_draws, 5), 'b--', 'LineWidth', 1.5);
xline(prctile(irf_draws, 95), 'b--', 'LineWidth', 1.5);
title('Distribution: MP Shock → Output (h=10)');
xlabel('IRF value');
ylabel('Density');
grid on;

subplot(1, 2, 2);
plot(irf_draws, 'k.', 'MarkerSize', 8);
hold on;
yline(median(irf_draws), 'r-', 'LineWidth', 2);
yline(prctile(irf_draws, 5), 'b--', 'LineWidth', 1.5);
yline(prctile(irf_draws, 95), 'b--', 'LineWidth', 1.5);
title('Accepted Models: MP Shock → Output (h=10)');
xlabel('Model index');
ylabel('IRF value');
grid on;

sgtitle('Identification Uncertainty in Sign Restrictions');
fprintf('Identification uncertainty plot created (Figure 2)\n');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 11: Impact vs multi-period restrictions comparison
%% ========================================================================
disp('========================================');
disp('EXAMPLE 11: Impact vs multi-period sign restrictions');
disp('Comparing periods_sign = 1 vs periods_sign = 4');
disp('========================================');

EQ_impact = signrestSVAR(data, nlags, sign_matrix, 1, var_names, ...
                         shock_names, 40, 500, true);
EQ_multi = signrestSVAR(data, nlags, sign_matrix, 4, var_names, ...
                        shock_names, 40, 500, true);
close all;

figure('Position', [100 100 1200 400]);

% Output response
subplot(1, 3, 1);
x_fill = [1:40, fliplr(1:40)];
y_fill_impact = [EQ_impact.Output_MonetaryPolicy(1,:), ...
                 fliplr(EQ_impact.Output_MonetaryPolicy(3,:))];
y_fill_multi = [EQ_multi.Output_MonetaryPolicy(1,:), ...
                fliplr(EQ_multi.Output_MonetaryPolicy(3,:))];
fill(x_fill, y_fill_impact, [0.8 0.8 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
hold on;
fill(x_fill, y_fill_multi, [0.5 0.5 1], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
plot(1:40, EQ_impact.Output_MonetaryPolicy(2,:), 'k-', 'LineWidth', 2, ...
     'DisplayName', 'Impact only');
plot(1:40, EQ_multi.Output_MonetaryPolicy(2,:), 'b--', 'LineWidth', 2, ...
     'DisplayName', 'Multi-period (4)');
plot(1:40, zeros(1,40), 'k-', 'LineWidth', 0.5);
xline(4, 'r--', 'LineWidth', 1, 'HandleVisibility', 'off');
text(4.5, min(ylim), 'Restrictions end', 'Color', 'r');
title('Output Response');
xlabel('Horizon');
ylabel('Response');
legend('Location', 'best');
grid on;

% Inflation response
subplot(1, 3, 2);
fill(x_fill, [EQ_impact.Inflation_MonetaryPolicy(1,:), ...
              fliplr(EQ_impact.Inflation_MonetaryPolicy(3,:))], ...
     [0.8 0.8 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
hold on;
fill(x_fill, [EQ_multi.Inflation_MonetaryPolicy(1,:), ...
              fliplr(EQ_multi.Inflation_MonetaryPolicy(3,:))], ...
     [0.5 0.5 1], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
plot(1:40, EQ_impact.Inflation_MonetaryPolicy(2,:), 'k-', 'LineWidth', 2, ...
     'DisplayName', 'Impact only');
plot(1:40, EQ_multi.Inflation_MonetaryPolicy(2,:), 'b--', 'LineWidth', 2, ...
     'DisplayName', 'Multi-period (4)');
plot(1:40, zeros(1,40), 'k-', 'LineWidth', 0.5);
xline(4, 'r--', 'LineWidth', 1, 'HandleVisibility', 'off');
title('Inflation Response');
xlabel('Horizon');
ylabel('Response');
legend('Location', 'best');
grid on;

% Interest rate response
subplot(1, 3, 3);
fill(x_fill, [EQ_impact.InterestRate_MonetaryPolicy(1,:), ...
              fliplr(EQ_impact.InterestRate_MonetaryPolicy(3,:))], ...
     [0.8 0.8 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
hold on;
fill(x_fill, [EQ_multi.InterestRate_MonetaryPolicy(1,:), ...
              fliplr(EQ_multi.InterestRate_MonetaryPolicy(3,:))], ...
     [0.5 0.5 1], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
plot(1:40, EQ_impact.InterestRate_MonetaryPolicy(2,:), 'k-', 'LineWidth', 2, ...
     'DisplayName', 'Impact only');
plot(1:40, EQ_multi.InterestRate_MonetaryPolicy(2,:), 'b--', 'LineWidth', 2, ...
     'DisplayName', 'Multi-period (4)');
plot(1:40, zeros(1,40), 'k-', 'LineWidth', 0.5);
xline(4, 'r--', 'LineWidth', 1, 'HandleVisibility', 'off');
title('Interest Rate Response');
xlabel('Horizon');
ylabel('Response');
legend('Location', 'best');
grid on;

sgtitle('Impact vs Multi-Period Sign Restrictions');
fprintf('Comparison plot created (Figure 3)\n');
fprintf('\n');

disp('Note: Multi-period restrictions produce tighter identification');
disp('but require restrictions to hold longer');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 12: Verbose option
%% ========================================================================
disp('========================================');
disp('EXAMPLE 12: Using verbose parameter');
disp('With verbose=true, iteration numbers are printed');
disp('========================================');

disp('Running with verbose=true (will print iterations)...');
EQ12 = signrestSVAR(data, nlags, sign_matrix, 1, var_names, shock_names, ...
                    40, 100, true, [], [], [], [], 5, 0, true);
close all;

disp(' ');
disp('With verbose=false (default), no iteration numbers printed');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 13: Understanding the sign restriction matrix
%% ========================================================================
disp('========================================');
disp('EXAMPLE 13: Understanding the sign restriction matrix');
disp('========================================');

disp('Sign restriction matrix structure:');
disp('  Rows = Variables');
disp('  Columns = Shocks');
disp('  Values: 1 = positive, -1 = negative, 0 = no restriction');
fprintf('\n');

disp('Example: Monetary policy shock (3rd column)');
sign_example = [0   0  -1;
                0   0  -1;
                0   0   1];
disp(sign_example);
fprintf('\n');
disp('Interpretation:');
disp('  Monetary Policy shock must:');
disp('    - Decrease Output (row 1: -1)');
disp('    - Decrease Inflation (row 2: -1)');
disp('    - Increase Interest Rate (row 3: +1)');
fprintf('\n');

disp('Multiple shocks example:');
sign_multiple = [ 1   1  -1;
                 -1   1  -1;
                  0   0   1];
disp(sign_multiple);
fprintf('\n');
disp('Interpretation:');
disp('  Supply (col 1): Output ↑, Inflation ↓');
disp('  Demand (col 2): Output ↑, Inflation ↑');
disp('  Monetary (col 3): Output ↓, Inflation ↓, Rate ↑');
fprintf('\n\n');

%% ========================================================================
disp('========================================');
disp('All examples completed successfully!');
disp('========================================');
disp('Key Sign Restrictions SVAR features demonstrated:');
disp('- Basic sign restrictions (impact only)');
disp('- Multi-period sign restrictions');
disp('- Multiple shocks identified simultaneously');
disp('- Variance decomposition');
disp('- Custom shock sizes and confidence levels');
disp('- Seasonal dummies and exogenous variables');
disp('- Restricted reduced-form VAR');
disp('- Identification uncertainty analysis');
disp('- Impact vs multi-period comparison');
disp('- Verbose option for debugging');
disp('- Understanding sign restriction matrices');
disp(' ');
disp('Key concepts:');
disp('- Sign restrictions identify a SET of models, not a unique model');
disp('- Reported IRFs are MEDIAN across accepted models');
disp('- Focus on IDENTIFICATION uncertainty, not parameter uncertainty');
disp('- Can identify multiple shocks with different sign patterns');