%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% EXAMPLES FOR maxshareSVAR
%% Author: Lorenzo Menna
%% This script demonstrates all features of the maxshareSVAR function
%% (Max share identification - Uhlig 2003, Barsky-Sims 2011, Francis et al. 2014)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all; clc;

%% ========================================================================
%% DATA SIMULATION
%% ========================================================================
% We simulate a 3-variable VAR(2) system for max share identification

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
disp('CRITICAL: Variable ordering matters for max share!');
disp('========================================');
fprintf('\n');

%% ========================================================================
%% EXAMPLE 1: Basic max share SVAR (cumulative variance, H=20)
%% ========================================================================
disp('========================================');
disp('EXAMPLE 1: Max share SVAR - identifying TFP shock');
disp('First variable = Output (whose variance we maximize)');
disp('========================================');

% IMPORTANT: Put the variable of interest FIRST!
var_names = {'Output', 'Hours', 'Inflation'};
shock_names = {'TFP', 'LaborSupply', 'MonetaryPolicy'};

H = 20;        % Maximize variance at 20 quarters (medium-run)
accum = 1;     % Cumulative variance (default)

EQ1 = maxshareSVAR(data, H, nlags, accum, var_names, shock_names, ...
                   40, 1000, true);
close all;

disp('Estimated impact matrix B:');
disp(EQ1.B);
fprintf('\n');

disp('Variance decomposition of Output at selected horizons:');
disp('Horizon    TFP        LaborSupply   MonetaryPolicy');
horizons = [1, 4, 20, 40];
for h = horizons
    fprintf('%3d        %.2f%%      %.2f%%          %.2f%%\n', h, ...
            EQ1.vardecshock_TFP(1,h), ...
            EQ1.vardecshock_LaborSupply(1,h), ...
            EQ1.vardecshock_MonetaryPolicy(1,h));
end
fprintf('\n');
fprintf('Note: TFP shock explains maximum variance at H=%d (cumulative)\n', H);
fprintf('TFP contribution at H=%d: %.2f%%\n', H, EQ1.vardecshock_TFP(1,H));
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 2: Effect of horizon H on identification
%% ========================================================================
disp('========================================');
disp('EXAMPLE 2: Comparing different horizons H');
disp('Short-run (H=1), Medium-run (H=20), Long-run (H=40)');
disp('========================================');

EQ_H1 = maxshareSVAR(data, 1, nlags, 1, var_names, shock_names, ...
                     40, 1000, true);
EQ_H20 = maxshareSVAR(data, 20, nlags, 1, var_names, shock_names, ...
                      40, 1000, true);
EQ_H40 = maxshareSVAR(data, 40, nlags, 1, var_names, shock_names, ...
                      40, 1000, true);
close all;

disp('TFP shock variance contribution to Output:');
disp('Horizon    H=1       H=20      H=40');
for h = [1, 4, 20, 40]
    fprintf('%3d        %.2f%%    %.2f%%    %.2f%%\n', h, ...
            EQ_H1.vardecshock_TFP(1,h), ...
            EQ_H20.vardecshock_TFP(1,h), ...
            EQ_H40.vardecshock_TFP(1,h));
end
fprintf('\n');
disp('Note: Different H identifies different shocks!');
disp('H=1: Impact shock (most important on impact)');
disp('H=20: Medium-run shock (business cycle frequencies)');
disp('H=40: Long-run shock (permanent component)');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 3: Cumulative (accum=1) vs Point (accum=0) variance
%% ========================================================================
disp('========================================');
disp('EXAMPLE 3: Cumulative vs Point-in-time variance maximization');
disp('========================================');

H = 20;
EQ_cumul = maxshareSVAR(data, H, nlags, 1, var_names, shock_names, ...
                        40, 1000, true);  % accum=1
EQ_point = maxshareSVAR(data, H, nlags, 0, var_names, shock_names, ...
                        40, 1000, true);  % accum=0
close all;

disp('TFP shock variance contribution to Output:');
disp('Horizon    Cumulative   Point-in-time');
for h = [1, 10, 20, 30, 40]
    fprintf('%3d        %.2f%%        %.2f%%\n', h, ...
            EQ_cumul.vardecshock_TFP(1,h), ...
            EQ_point.vardecshock_TFP(1,h));
end
fprintf('\n');
fprintf('At H=%d:\n', H);
fprintf('  Cumulative (accum=1): TFP explains %.2f%% of cumulative variance\n', ...
        EQ_cumul.vardecshock_TFP(1,H));
fprintf('  Point (accum=0):      TFP explains %.2f%% of cumulative variance (0 to %d)\n', ...
        EQ_point.vardecshock_TFP(1,H), H);
fprintf('\n');
disp('IMPORTANT: Reported variance decomposition is ALWAYS cumulative!');
disp('With accum=0, the shock maximizes variance at exactly H=20,');
disp('where it explains 100% of the non-cumulative variance.');
disp('But the cumulative variance decomposition (reported here) includes');
disp('all horizons 0 to H, so it will be less than 100%.');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 4: Variable ordering matters!
%% ========================================================================
disp('========================================');
disp('EXAMPLE 4: Demonstrating that variable ordering matters');
disp('========================================');

% Original ordering: Output first
data_order1 = data;
var_names_order1 = {'Output', 'Hours', 'Inflation'};
shock_names_order1 = {'TFP', 'Shock2', 'Shock3'};

% Alternative ordering: Hours first
data_order2 = data(:, [2, 1, 3]);  % Reorder columns
var_names_order2 = {'Hours', 'Output', 'Inflation'};
shock_names_order2 = {'LaborSupply', 'Shock2', 'Shock3'};

H = 20;
EQ_order1 = maxshareSVAR(data_order1, H, nlags, 1, var_names_order1, ...
                         shock_names_order1, 40, 1000, true);
EQ_order2 = maxshareSVAR(data_order2, H, nlags, 1, var_names_order2, ...
                         shock_names_order2, 40, 1000, true);
close all;

disp('Order 1: Output first → TFP shock identified');
fprintf('  TFP explains %.2f%% of Output variance at H=%d\n', ...
        EQ_order1.vardecshock_TFP(1,H), H);

disp('Order 2: Hours first → Labor Supply shock identified');
fprintf('  LaborSupply explains %.2f%% of Hours variance at H=%d\n', ...
        EQ_order2.vardecshock_LaborSupply(1,H), H);

fprintf('\n');
disp('CRITICAL: Different orderings identify different shocks!');
disp('Place the variable of interest in the FIRST position.');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 5: Custom shock sizes
%% ========================================================================
disp('========================================');
disp('EXAMPLE 5: Max share SVAR with custom shock sizes');
disp('========================================');

shock_size = 2 * diag(EQ1.B);  % 2 std deviations
EQ3 = maxshareSVAR(data, H, nlags, accum, var_names, shock_names, ...
                   40, 1000, true, [], [], [], shock_size);
close all;

disp('Comparison of IRF magnitudes at impact (horizon 2):');
disp('           1-SD shock    2-SD shock');
fprintf('Output:    %.4f        %.4f\n', ...
        EQ1.Output_TFP(2,2), EQ3.Output_TFP(2,2));
fprintf('Expected ratio: ~2.0, Actual ratio: %.4f\n', ...
        EQ3.Output_TFP(2,2) / EQ1.Output_TFP(2,2));
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 6: Different confidence levels
%% ========================================================================
disp('========================================');
disp('EXAMPLE 6: Max share SVAR with different confidence intervals');
disp('========================================');

EQ4a = maxshareSVAR(data, H, nlags, accum, var_names, shock_names, ...
                    40, 1000, true, [], [], [], [], 5);    % 90% CI
EQ4b = maxshareSVAR(data, H, nlags, accum, var_names, shock_names, ...
                    40, 1000, true, [], [], [], [], 16);   % 68% CI
close all;

disp('Output response to TFP shock at horizon 10:');
fprintf('90%% CI: [%.4f, %.4f]  Width: %.4f\n', ...
        EQ4a.Output_TFP(3,10), EQ4a.Output_TFP(1,10), ...
        EQ4a.Output_TFP(1,10) - EQ4a.Output_TFP(3,10));
fprintf('68%% CI: [%.4f, %.4f]  Width: %.4f\n', ...
        EQ4b.Output_TFP(3,10), EQ4b.Output_TFP(1,10), ...
        EQ4b.Output_TFP(1,10) - EQ4b.Output_TFP(3,10));
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 7: With seasonal dummies
%% ========================================================================
disp('========================================');
disp('EXAMPLE 7: Max share SVAR with quarterly seasonal dummies');
disp('========================================');

EQ5 = maxshareSVAR(data, H, nlags, accum, var_names, shock_names, ...
                   40, 1000, true, [], 'quarter');
close all;

disp('Seasonal patterns controlled for in estimation');
disp('TFP shock maximizes deseasonalized Output variance');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 8: With exogenous variables
%% ========================================================================
disp('========================================');
disp('EXAMPLE 8: Max share SVAR with exogenous oil price shock');
disp('========================================');

% Create oil price shock dummy
exog = zeros(T, 1);
exog(80:85) = 1;  % Oil shock period

% Add oil shock effect to data
data_oil = data;
oil_effect = B_true * [0; 0.5; 1];  % Affects hours and inflation
for t = 80:85
    data_oil(t, :) = data_oil(t, :) + oil_effect';
end

EQ6 = maxshareSVAR(data_oil, H, nlags, accum, var_names, shock_names, ...
                   40, 1000, true, [], [], exog);
close all;

disp('Oil price shock controlled for via exogenous dummy');
disp('TFP shock maximizes Output variance net of oil effects');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 9: With restricted reduced-form VAR
%% ========================================================================
disp('========================================');
disp('EXAMPLE 9: Max share SVAR with restricted reduced-form VAR');
disp('========================================');

lr = [1 1 0;   % Output affected by output, hours only
      1 1 0;   % Hours affected by output, hours only
      1 1 1];  % Inflation affected by all variables

EQ7 = maxshareSVAR(data, H, nlags, accum, var_names, shock_names, ...
                   40, 1000, true, lr);
close all;

disp('Reduced-form restrictions on lags imposed via lr matrix');
disp('Max share identification applied to restricted VAR');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 10: Structural shocks analysis
%% ========================================================================
disp('========================================');
disp('EXAMPLE 10: Structural shocks from max share identification');
disp('========================================');

% Structural shocks should be orthogonal
shocks = EQ1.struc;
corr_matrix = corr(shocks');

disp('Correlation matrix of structural shocks:');
disp('(Should be approximately identity matrix)');
disp(corr_matrix);

% Summary statistics
disp('Structural shock statistics:');
disp('           Mean       Std Dev');
for i = 1:K
    fprintf('%-15s %.4f     %.4f\n', shock_names{i}, ...
            mean(shocks(i,:)), std(shocks(i,:)));
end

% Plot TFP shock (the identified shock)
figure('Position', [100 100 1000 400]);
plot(shocks(1,:), 'k-', 'LineWidth', 1.5);
title('TFP Shock (Max Share Identification)');
ylabel('Magnitude');
xlabel('Time');
grid on;
fprintf('TFP shock plot created (Figure 1)\n');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 11: Variance decomposition visualization
%% ========================================================================
disp('========================================');
disp('EXAMPLE 11: Variance decomposition showing max share property');
disp('========================================');

% Plot variance decomposition for Output
figure('Position', [100 100 1200 400]);

subplot(1, 3, 1);
plot(1:40, EQ1.vardecshock_TFP(1,:), 'k-', 'LineWidth', 2);
hold on;
plot(1:40, EQ1.vardecshock_LaborSupply(1,:), 'k--', 'LineWidth', 2);
plot(1:40, EQ1.vardecshock_MonetaryPolicy(1,:), 'k:', 'LineWidth', 2);
xline(H, 'r--', 'LineWidth', 1.5);
text(H, 80, sprintf('H=%d', H), 'Color', 'r', 'FontSize', 12);
title('Output Variance Decomposition');
xlabel('Horizon');
ylabel('Percentage');
legend('TFP (max share)', 'LaborSupply', 'MonetaryPolicy', 'Location', 'best');
ylim([0 105]);
grid on;

subplot(1, 3, 2);
plot(1:40, EQ1.vardecshock_TFP(2,:), 'k-', 'LineWidth', 2);
hold on;
plot(1:40, EQ1.vardecshock_LaborSupply(2,:), 'k--', 'LineWidth', 2);
plot(1:40, EQ1.vardecshock_MonetaryPolicy(2,:), 'k:', 'LineWidth', 2);
title('Hours Variance Decomposition');
xlabel('Horizon');
ylabel('Percentage');
legend('TFP', 'LaborSupply', 'MonetaryPolicy', 'Location', 'best');
ylim([0 105]);
grid on;

subplot(1, 3, 3);
plot(1:40, EQ1.vardecshock_TFP(3,:), 'k-', 'LineWidth', 2);
hold on;
plot(1:40, EQ1.vardecshock_LaborSupply(3,:), 'k--', 'LineWidth', 2);
plot(1:40, EQ1.vardecshock_MonetaryPolicy(3,:), 'k:', 'LineWidth', 2);
title('Inflation Variance Decomposition');
xlabel('Horizon');
ylabel('Percentage');
legend('TFP', 'LaborSupply', 'MonetaryPolicy', 'Location', 'best');
ylim([0 105]);
grid on;

sgtitle(sprintf('Max Share at H=%d (cumulative)', H));
fprintf('Variance decomposition plot created (Figure 2)\n');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 12: Only first shock is identified
%% ========================================================================
disp('========================================');
disp('EXAMPLE 12: Understanding partial identification');
disp('Only the first shock (TFP) is identified; others are not unique');
disp('========================================');

disp('Key point: Max share identifies ONLY the first shock!');
disp('The remaining K-1 shocks are orthogonal to the first shock,');
disp('but their ordering is arbitrary (rotational indeterminacy).');
fprintf('\n');

% Show that we typically only analyze the first shock
disp('Typical usage: Focus on first shock only');
disp('IRF fields available:');
fprintf('  EQ.Output_TFP           ← Analyze this (identified)\n');
fprintf('  EQ.Output_LaborSupply   ← Usually ignore (not identified)\n');
fprintf('  EQ.Output_MonetaryPolicy← Usually ignore (not identified)\n');
fprintf('\n');

% Show IRF of first shock only
figure('Position', [100 100 800 600]);
for i = 1:K
    subplot(K, 1, i);
    
    % Extract IRF data for first shock only
    eval(['irf_data = EQ1.' var_names{i} '_' shock_names{1} ';']);
    
    % Plot confidence bands
    x_fill = [1:40, fliplr(1:40)];
    y_fill = [irf_data(1,:), fliplr(irf_data(3,:))];
    fill(x_fill, y_fill, [0.8 0.8 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
    hold on;
    
    % Plot point estimate
    plot(1:40, irf_data(2,:), 'k-', 'LineWidth', 2);
    plot(1:40, zeros(1,40), 'k--', 'LineWidth', 0.5);
    
    title([var_names{i} ' response to ' shock_names{1} ' shock']);
    xlabel('Horizon');
    ylabel('Response');
    grid on;
end
sgtitle('Max Share SVAR: Only First Shock (TFP) is Analyzed');
fprintf('First shock IRF visualization created (Figure 3)\n');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 13: Comparison of H values for TFP identification
%% ========================================================================
disp('========================================');
disp('EXAMPLE 13: Choosing the right horizon H for TFP shocks');
disp('========================================');

% Estimate with different H values
H_values = [1, 4, 12, 20, 40];

% Pre-compute all estimations (suppress figures by storing results first)
disp('Estimating for different H values...');
EQ_H = cell(length(H_values), 1);
for idx = 1:length(H_values)
    H_test = H_values(idx);
    fprintf('  H = %d\n', H_test);
    EQ_H{idx} = maxshareSVAR(data, H_test, nlags, 1, var_names, shock_names, ...
                             40, 500, true);
end
close all;  % Close all the automatic figures AFTER estimation

% Now create the comparison plot
figure('Position', [100 100 1400 500]);

% Left panel: IRF comparison
subplot(1, 2, 1);
colors = lines(length(H_values));  % Get distinct colors
for idx = 1:length(H_values)
    H_test = H_values(idx);
    plot(1:40, EQ_H{idx}.Output_TFP(2,:), 'LineWidth', 2, 'Color', colors(idx,:), ...
         'DisplayName', sprintf('H=%d', H_test));
    hold on;
end
plot(1:40, zeros(1,40), 'k--', 'LineWidth', 0.5);
title('Output Response to TFP Shock (Different H)');
xlabel('Horizon');
ylabel('Response');
legend('Location', 'best');
grid on;

% Right panel: Variance decomposition comparison
subplot(1, 2, 2);
for idx = 1:length(H_values)
    H_test = H_values(idx);
    plot(1:40, EQ_H{idx}.vardecshock_TFP(1,:), 'LineWidth', 2, 'Color', colors(idx,:), ...
         'DisplayName', sprintf('H=%d', H_test));
    hold on;
    
    % Mark the horizon H where variance is maximized
    xline(H_test, ':', 'Color', colors(idx,:), 'LineWidth', 1, 'HandleVisibility', 'off');
end
title('TFP Contribution to Output Variance (Different H)');
xlabel('Horizon');
ylabel('Percentage');
legend('Location', 'best');
ylim([0 105]);
grid on;

sgtitle('Effect of Horizon H on TFP Shock Identification');
fprintf('Horizon comparison plot created (Figure 4)\n');
fprintf('\n');

% Print variance contributions at each H
disp('TFP variance contribution to Output at horizon H:');
disp('H      Variance at H');
for idx = 1:length(H_values)
    H_test = H_values(idx);
    fprintf('%2d     %.2f%%\n', H_test, EQ_H{idx}.vardecshock_TFP(1, H_test));
end
fprintf('\n');

disp('Guidance for choosing H:');
disp('  H=1:    Business cycle shock (impact maximizer)');
disp('  H=4-8:  Short-run shock');
disp('  H=12-20: Medium-run/business cycle shock (typical for TFP)');
disp('  H=40+:  Long-run/permanent shock');
fprintf('\n\n');

%% ========================================================================
disp('========================================');
disp('All examples completed successfully!');
disp('========================================');
disp('Key Max Share SVAR features demonstrated:');
disp('- Variable ordering matters (first variable is special)');
disp('- Only first shock is identified (partial identification)');
disp('- Horizon H determines which shock is identified');
disp('- Cumulative (accum=1) vs point (accum=0) variance maximization');
disp('- Max share property: largest variance contribution at horizon H');
disp('- Custom shock sizes and confidence levels');
disp('- Seasonal dummies and exogenous variables');
disp('- Restricted reduced-form VAR');
disp('- Structural shock extraction');
disp('- Variance decomposition analysis');
disp('- Comparison across different H values');