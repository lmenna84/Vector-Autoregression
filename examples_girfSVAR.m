%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% EXAMPLES FOR girfSVAR
%% Author: Lorenzo Menna
%% This script demonstrates all features of the girfSVAR function
%% (Generalized Impulse Response Functions - Pesaran and Shin 1998)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all; clc;

%% ========================================================================
%% DATA SIMULATION
%% ========================================================================
% We simulate a 3-variable VAR(2) system to demonstrate GIRFs

rng(12345);  % Set seed for reproducibility

% Parameters
K = 3;        % Number of variables
T = 200;      % Sample size
nlags = 2;    % Number of lags

% True VAR coefficient matrices (stable system)
A1 = [0.5  0.1  0.2;
      0.1  0.6  0.1;
      0.2  0.1  0.4];

A2 = [0.2  0.0  0.1;
      0.0  0.1  0.0;
      0.1  0.0  0.2];

nu = [0.5; 1.0; 0.2];  % Constants

% True impact matrix (NOT necessarily triangular for GIRFs)
B_true = [1.0  0.3  0.2;
          0.5  1.2  0.1;
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
disp('GIRFs are order-invariant and variance-normalized');
disp('========================================');
fprintf('\n');

%% ========================================================================
%% EXAMPLE 1: Basic generalized impulse response functions
%% ========================================================================
disp('========================================');
disp('EXAMPLE 1: Basic GIRFs');
disp('Generalized IRFs (Pesaran and Shin 1998)');
disp('========================================');

var_names = {'Output', 'Inflation', 'InterestRate'};
shock_names = {'OutputShock', 'InflationShock', 'RateShock'};

EQ1 = girfSVAR(data, nlags, var_names, shock_names);
close all;

disp('Key features of GIRFs:');
disp('  - Order-invariant (not affected by variable ordering)');
disp('  - Each shock normalized to 1 SD of that variable');
disp('  - Non-orthogonal shocks (allow for correlations)');
disp('  - Variance decomposition may not sum to 100%');
fprintf('\n');

disp('Variance-covariance matrix:');
disp(EQ1.P);
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 2: Order invariance demonstration
%% ========================================================================
disp('========================================');
disp('EXAMPLE 2: Order invariance - key advantage of GIRFs');
disp('GIRFs give same results regardless of variable ordering');
disp('========================================');

% Original ordering
var_names1 = {'Output', 'Inflation', 'InterestRate'};
shock_names1 = {'OutputShock', 'InflationShock', 'RateShock'};
data1 = data;

EQ_order1 = girfSVAR(data1, nlags, var_names1, shock_names1);
close all;

% Different ordering: Interest Rate first
var_names2 = {'InterestRate', 'Output', 'Inflation'};
shock_names2 = {'RateShock', 'OutputShock', 'InflationShock'};
data2 = data(:, [3, 1, 2]);  % Reorder columns

EQ_order2 = girfSVAR(data2, nlags, var_names2, shock_names2);
close all;

% Compare responses (Output response to Inflation shock)
disp('Output response to Inflation shock at horizon 10:');
fprintf('  Original ordering:  %.6f\n', EQ_order1.Output_InflationShock(2,10));
fprintf('  Different ordering: %.6f\n', EQ_order2.Output_InflationShock(2,10));
fprintf('  Difference:         %.10f (should be ~0)\n', ...
        abs(EQ_order1.Output_InflationShock(2,10) - EQ_order2.Output_InflationShock(2,10)));
fprintf('\n');

disp('GIRFs are ORDER-INVARIANT - unlike Cholesky decomposition');
disp('This is a major advantage when variable ordering is unclear');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 3: Understanding variance normalization
%% ========================================================================
disp('========================================');
disp('EXAMPLE 3: Variance normalization in GIRFs');
disp('Each shock is normalized to 1 SD of that variable');
disp('========================================');

disp('Standard deviations of reduced-form innovations:');
for i = 1:K
    fprintf('  %s: %.4f\n', var_names{i}, sqrt(EQ1.P(i,i)));
end
fprintf('\n');

disp('GIRF normalization:');
disp('  Each shock represents a 1-SD increase in that variable''s innovation');
disp('  This makes shocks comparable across variables');
fprintf('\n');

% Compare impact responses
disp('Impact responses (horizon 1):');
for i = 1:K
    for j = 1:K
        fprintf('  %s → %s: %.4f\n', shock_names{j}, var_names{i}, ...
                eval(['EQ1.' var_names{i} '_' shock_names{j} '(2,2)']));
    end
end
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 4: Non-orthogonal shocks and variance decomposition
%% ========================================================================
disp('========================================');
disp('EXAMPLE 4: Non-orthogonal shocks in GIRFs');
disp('Variance decomposition may exceed 100%');
disp('========================================');

disp('Variance decomposition of Output at various horizons:');
disp('Horizon    OutputShock  InflShock  RateShock    Total');
for h = [1, 4, 12, 20, 40]
    total = EQ1.vardecshock_OutputShock(1,h) + ...
            EQ1.vardecshock_InflationShock(1,h) + ...
            EQ1.vardecshock_RateShock(1,h);
    fprintf('%3d        %.2f%%       %.2f%%     %.2f%%      %.2f%%\n', h, ...
            EQ1.vardecshock_OutputShock(1,h), ...
            EQ1.vardecshock_InflationShock(1,h), ...
            EQ1.vardecshock_RateShock(1,h), ...
            total);
end
fprintf('\n');

disp('Note: Total may not equal 100% because shocks are non-orthogonal');
disp('This reflects actual correlations between innovations');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 5: Variance decomposition visualization
%% ========================================================================
disp('========================================');
disp('EXAMPLE 5: Variance decomposition with GIRFs');
disp('========================================');

% Plot variance decomposition
figure('Position', [100 100 1200 400]);

subplot(1, 3, 1);
plot(1:40, EQ1.vardecshock_OutputShock(1,:), 'k-', 'LineWidth', 2);
hold on;
plot(1:40, EQ1.vardecshock_InflationShock(1,:), 'k--', 'LineWidth', 2);
plot(1:40, EQ1.vardecshock_RateShock(1,:), 'k:', 'LineWidth', 2);
plot(1:40, 100*ones(1,40), 'r--', 'LineWidth', 1);
title('Output Variance Decomposition');
xlabel('Horizon');
ylabel('Percentage');
legend('Output Shock', 'Inflation Shock', 'Rate Shock', '100%', 'Location', 'best');
ylim([0 120]);
grid on;

subplot(1, 3, 2);
plot(1:40, EQ1.vardecshock_OutputShock(2,:), 'k-', 'LineWidth', 2);
hold on;
plot(1:40, EQ1.vardecshock_InflationShock(2,:), 'k--', 'LineWidth', 2);
plot(1:40, EQ1.vardecshock_RateShock(2,:), 'k:', 'LineWidth', 2);
plot(1:40, 100*ones(1,40), 'r--', 'LineWidth', 1);
title('Inflation Variance Decomposition');
xlabel('Horizon');
ylabel('Percentage');
legend('Output Shock', 'Inflation Shock', 'Rate Shock', '100%', 'Location', 'best');
ylim([0 120]);
grid on;

subplot(1, 3, 3);
plot(1:40, EQ1.vardecshock_OutputShock(3,:), 'k-', 'LineWidth', 2);
hold on;
plot(1:40, EQ1.vardecshock_InflationShock(3,:), 'k--', 'LineWidth', 2);
plot(1:40, EQ1.vardecshock_RateShock(3,:), 'k:', 'LineWidth', 2);
plot(1:40, 100*ones(1,40), 'r--', 'LineWidth', 1);
title('Interest Rate Variance Decomposition');
xlabel('Horizon');
ylabel('Percentage');
legend('Output Shock', 'Inflation Shock', 'Rate Shock', '100%', 'Location', 'best');
ylim([0 120]);
grid on;

sgtitle('Variance Decomposition (GIRFs - may exceed 100%)');
fprintf('Variance decomposition plot created (Figure 1)\n');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 6: Different confidence levels
%% ========================================================================
disp('========================================');
disp('EXAMPLE 6: GIRFs with different confidence intervals');
disp('========================================');

EQ2a = girfSVAR(data, nlags, var_names, shock_names, 40, 1000, true, [], [], [], 5);    % 90% CI
EQ2b = girfSVAR(data, nlags, var_names, shock_names, 40, 1000, true, [], [], [], 16);   % 68% CI
close all;

disp('Output response to Inflation shock at horizon 10:');
fprintf('90%% CI: [%.4f, %.4f]  Width: %.4f\n', ...
        EQ2a.Output_InflationShock(3,10), EQ2a.Output_InflationShock(1,10), ...
        EQ2a.Output_InflationShock(1,10) - EQ2a.Output_InflationShock(3,10));
fprintf('68%% CI: [%.4f, %.4f]  Width: %.4f\n', ...
        EQ2b.Output_InflationShock(3,10), EQ2b.Output_InflationShock(1,10), ...
        EQ2b.Output_InflationShock(1,10) - EQ2b.Output_InflationShock(3,10));
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 7: With seasonal dummies
%% ========================================================================
disp('========================================');
disp('EXAMPLE 7: GIRFs with quarterly seasonal dummies');
disp('========================================');

EQ3 = girfSVAR(data, nlags, var_names, shock_names, 40, 1000, true, [], 'quarter');
close all;

disp('Seasonal patterns controlled for in estimation');
disp('GIRFs computed from deseasonalized innovations');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 8: With exogenous variables
%% ========================================================================
disp('========================================');
disp('EXAMPLE 8: GIRFs with exogenous oil price shock');
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

EQ4 = girfSVAR(data_oil, nlags, var_names, shock_names, 40, 1000, true, [], [], exog);
close all;

disp('Oil price shock controlled for via exogenous dummy');
disp('GIRFs computed from oil-adjusted innovations');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 9: With restricted reduced-form VAR
%% ========================================================================
disp('========================================');
disp('EXAMPLE 9: GIRFs with restricted reduced-form VAR');
disp('========================================');

lr = [1 1 0;   % Output affected by output, inflation only
      1 1 0;   % Inflation affected by output, inflation only
      1 1 1];  % Interest rate affected by all variables

EQ5 = girfSVAR(data, nlags, var_names, shock_names, 40, 1000, true, lr);
close all;

disp('Reduced-form restrictions on lags imposed via lr matrix');
disp('GIRFs computed from restricted VAR');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 10: Understanding generalized structural shocks
%% ========================================================================
disp('========================================');
disp('EXAMPLE 10: Generalized structural shocks (non-orthogonal)');
disp('========================================');

disp('EQ.struc contains generalized structural shocks');
fprintf('Size: %dx%d (%d shocks × %d effective observations)\n', ...
        size(EQ1.struc, 1), size(EQ1.struc, 2), K, T-nlags);
fprintf('\n');

% Check orthogonality
shock_corr = corrcoef(EQ1.struc');
disp('Correlation matrix of generalized structural shocks:');
disp(shock_corr);
fprintf('\n');

disp('Note: Shocks are NOT orthogonal (off-diagonal elements ≠ 0)');
disp('This reflects actual correlations in the data');
fprintf('\n');

% Plot generalized structural shocks
T_effective = T - nlags;
figure('Position', [100 100 1200 800]);
for i = 1:K
    subplot(K, 1, i);
    plot(1:T_effective, EQ1.struc(i,:), 'k-', 'LineWidth', 1);
    hold on;
    plot(1:T_effective, zeros(1,T_effective), 'r--', 'LineWidth', 0.5);
    title([shock_names{i} ' (generalized structural shock)']);
    xlabel('Time');
    ylabel('Shock');
    grid on;
end
sgtitle('Generalized Structural Shocks (Non-Orthogonal)');
fprintf('Structural shocks plot created (Figure 2)\n');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 11: Monte Carlo uncertainty analysis
%% ========================================================================
disp('========================================');
disp('EXAMPLE 11: Examining Monte Carlo uncertainty (EQ.irf_sim)');
disp('========================================');

% Extract all Monte Carlo draws for Inflation shock effect on Output at h=10
% irf_sim dimensions: [variable, horizon, repetition, shock]
irf_draws = squeeze(EQ1.irf_sim(1, 10, :, 2));  % Output, h=10, all reps, Inflation shock

disp('Distribution of GIRFs across Monte Carlo draws:');
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
xline(median(irf_draws), 'r-', 'LineWidth', 2, 'Label', 'Median');
xline(prctile(irf_draws, 5), 'b--', 'LineWidth', 1.5);
xline(prctile(irf_draws, 95), 'b--', 'LineWidth', 1.5);
title('Distribution: Inflation Shock → Output (h=10)');
xlabel('GIRF value');
ylabel('Density');
grid on;

subplot(1, 2, 2);
plot(irf_draws, 'k.', 'MarkerSize', 8);
hold on;
yline(median(irf_draws), 'r-', 'LineWidth', 2);
yline(prctile(irf_draws, 5), 'b--', 'LineWidth', 1.5);
yline(prctile(irf_draws, 95), 'b--', 'LineWidth', 1.5);
title('Monte Carlo Draws: Inflation Shock → Output (h=10)');
xlabel('Draw index');
ylabel('GIRF value');
grid on;

sgtitle('Monte Carlo Uncertainty in GIRFs');
fprintf('Monte Carlo uncertainty plot created (Figure 3)\n');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 12: Verbose option
%% ========================================================================
disp('========================================');
disp('EXAMPLE 12: Using verbose parameter');
disp('With verbose=true, Monte Carlo iteration numbers are printed');
disp('========================================');

disp('Running with verbose=true (will print iterations)...');
EQ6 = girfSVAR(data, nlags, var_names, shock_names, 40, 100, true, [], [], [], 5, true);
close all;

disp(' ');
disp('With verbose=false (default), no iteration numbers printed');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 13: When to use GIRFs
%% ========================================================================
disp('========================================');
disp('EXAMPLE 13: When to use GIRFs vs other identification methods');
disp('========================================');

disp('Use GIRFs when:');
disp('  1. Variable ordering is unclear or arbitrary');
disp('     (GIRFs are order-invariant)');
disp('  ');
disp('  2. You want to allow for correlated shocks');
disp('     (more realistic than orthogonal shocks)');
disp('  ');
disp('  3. You need variance-normalized shocks');
disp('     (each shock is 1 SD, making them comparable)');
disp('  ');
disp('  4. Economic theory does not suggest clear restrictions');
disp('     (no sign restrictions, no long-run restrictions)');
fprintf('\n');

disp('Do NOT use GIRFs when:');
disp('  1. You need unique structural identification');
disp('     (GIRFs do not identify structural shocks uniquely)');
disp('  ');
disp('  2. You have clear economic restrictions');
disp('     (use sign restrictions, long-run restrictions, etc.)');
disp('  ');
disp('  3. You need orthogonal shocks');
disp('     (use Cholesky, other structural methods)');
fprintf('\n');

disp('Key differences: GIRFs vs Cholesky');
disp('  GIRFs:     Order-invariant, non-orthogonal, variance-normalized');
disp('  Cholesky:  Order-dependent, orthogonal, arbitrary normalization');
fprintf('\n\n');

%% ========================================================================
disp('========================================');
disp('All examples completed successfully!');
disp('========================================');
disp('Key GIRF features demonstrated:');
disp('- Basic generalized impulse response functions');
disp('- Order invariance (major advantage)');
disp('- Variance normalization (1 SD per shock)');
disp('- Non-orthogonal shocks and variance decomposition');
disp('- Variance decomposition visualization');
disp('- Different confidence levels');
disp('- Seasonal dummies and exogenous variables');
disp('- Restricted reduced-form VAR');
disp('- Generalized structural shocks (non-orthogonal)');
disp('- Monte Carlo uncertainty analysis');
disp('- Verbose option for debugging');
disp('- When to use GIRFs vs other methods');
disp(' ');
disp('Key concepts:');
disp('- GIRFs are ORDER-INVARIANT (unlike Cholesky)');
disp('- Shocks are NON-ORTHOGONAL (reflect actual correlations)');
disp('- Each shock is 1-SD normalized (comparable across variables)');
disp('- Variance decomposition may EXCEED 100% (due to correlations)');
disp('- GIRFs do NOT uniquely identify structural shocks');