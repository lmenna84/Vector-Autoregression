%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% EXAMPLES FOR gmmLRSVAR
%% Author: Lorenzo Menna
%% This script demonstrates all features of the gmmLRSVAR function
%% (GMM-estimated Structural VAR with long-run restrictions)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all; clc;

%% ========================================================================
%% DATA SIMULATION
%% ========================================================================
% We simulate a 3-variable VAR(2) system with long-run restrictions

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

% True structural impact matrix (will be used with long-run restrictions)
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
disp('========================================');
fprintf('\n');

%% ========================================================================
%% EXAMPLE 1: Basic GMM SVAR with long-run restrictions (Blanchard-Quah)
%% ========================================================================
disp('========================================');
disp('EXAMPLE 1: GMM long-run SVAR (Blanchard-Quah identification)');
disp('========================================');

var_names = {'Output', 'Inflation', 'Unemployment'};
shock_names = {'Supply', 'Demand', 'Labor'};

% Long-run identification matrix (lower triangular restrictions)
% Demand shock has no long-run effect on Output
% Labor shock has no long-run effect on Output or Inflation
idmat_lr = [1  0  0;
            1  1  0;
            1  1  1];

EQ1 = gmmLRSVAR(data, 2, idmat_lr, var_names, shock_names, 40, 1000, true);
close all;

disp('Estimated impact matrix P (GMM long-run):');
disp(EQ1.P);
fprintf('\n');

% Display IRF
disp('IRF: Output response to Supply shock (first 10 periods):');
disp('Period    95% CI    Point    5% CI');
for i = 1:10
    fprintf('%3d    %7.4f  %7.4f  %7.4f\n', i, ...
            EQ1.Output_Supply(1,i), EQ1.Output_Supply(2,i), EQ1.Output_Supply(3,i));
end
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 2: Verifying long-run restrictions
%% ========================================================================
disp('========================================');
disp('EXAMPLE 2: Verifying long-run restrictions via cumulative IRFs');
disp('========================================');

% Cumulative IRFs approximate long-run effects
cum_supply_output = sum(EQ1.Output_Supply(2,:));
cum_demand_output = sum(EQ1.Output_Demand(2,:));
cum_labor_output = sum(EQ1.Output_Labor(2,:));
cum_labor_inflation = sum(EQ1.Inflation_Labor(2,:));

disp('Cumulative IRFs (sum over 40 periods):');
fprintf('Supply shock -> Output:    %.4f (permanent effect)\n', cum_supply_output);
fprintf('Demand shock -> Output:    %.4f (should be ~0)\n', cum_demand_output);
fprintf('Labor shock -> Output:     %.4f (should be ~0)\n', cum_labor_output);
fprintf('Labor shock -> Inflation:  %.4f (should be ~0)\n', cum_labor_inflation);
fprintf('\n');

disp('Note: Demand and Labor shocks have only temporary effects');
disp('on Output due to long-run restrictions. Only Supply shock');
disp('has permanent effects.');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 3: Overidentified GMM long-run SVAR
%% ========================================================================
disp('========================================');
disp('EXAMPLE 3: GMM long-run SVAR with overidentifying restrictions');
disp('========================================');

% Overidentified: 4 zero restrictions (one more than minimum 3)
% Additional restriction: Supply shock has no long-run effect on Unemployment
idmat_over = [1  0  0;    % Output: only Supply has LR effect
              1  1  0;    % Inflation: Supply and Demand have LR effects
              0  1  1];   % Unemployment: Demand and Labor have LR effects (NOT Supply)

EQ2 = gmmLRSVAR(data, 2, idmat_over, var_names, shock_names, 40, 1000, true);
close all;

disp('Overidentified model:');
fprintf('  Zero restrictions: %d (one extra restriction)\n', sum(idmat_over(:)==0));
fprintf('  Minimum required: %d\n', K*(K-1)/2);
fprintf('  Overidentifying restrictions: %d\n', sum(idmat_over(:)==0) - K*(K-1)/2);
disp('  Impact matrix P:');
disp(EQ2.P);
fprintf('\n');

% Verify the extra restriction (Supply -> Unemployment should be ~0 in LR)
cum_supply_unemp = sum(EQ2.Unemployment_Supply(2,:));
fprintf('Cumulative effect of Supply shock on Unemployment: %.4f (should be ~0)\n', ...
        cum_supply_unemp);
fprintf('This is the overidentifying restriction being tested.\n');
fprintf('\n');

% Verify standard restrictions also hold
cum_demand_output = sum(EQ2.Output_Demand(2,:));
cum_labor_output = sum(EQ2.Output_Labor(2,:));
cum_labor_inflation = sum(EQ2.Inflation_Labor(2,:));

fprintf('Standard long-run restrictions also verified:\n');
fprintf('  Demand -> Output: %.4f (should be ~0)\n', cum_demand_output);
fprintf('  Labor -> Output: %.4f (should be ~0)\n', cum_labor_output);
fprintf('  Labor -> Inflation: %.4f (should be ~0)\n', cum_labor_inflation);
fprintf('\n\n');
%% ========================================================================
%% EXAMPLE 4: Custom shock sizes
%% ========================================================================
disp('========================================');
disp('EXAMPLE 4: GMM long-run SVAR with 2-standard-deviation shocks');
disp('========================================');

shock_size = 2 * diag(EQ1.P);  % 2 std deviations
EQ3 = gmmLRSVAR(data, 2, idmat_lr, var_names, shock_names, 40, 1000, ...
                true, [], [], [], shock_size);
close all;

disp('Comparison of IRF magnitudes at horizon 1:');
disp('           1-SD shock    2-SD shock');
fprintf('Output:    %.4f        %.4f\n', ...
        EQ1.Output_Supply(2,1), EQ3.Output_Supply(2,1));
fprintf('Expected ratio: ~2.0, Actual ratio: %.4f\n', ...
        EQ3.Output_Supply(2,1) / EQ1.Output_Supply(2,1));
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 5: Different confidence levels
%% ========================================================================
disp('========================================');
disp('EXAMPLE 5: GMM long-run SVAR with different confidence intervals');
disp('Comparing 90% (default) vs 68% confidence bands');
disp('========================================');

EQ4a = gmmLRSVAR(data, 2, idmat_lr, var_names, shock_names, 40, 1000, ...
                 true, [], [], [], [], 5);    % 90% CI
EQ4b = gmmLRSVAR(data, 2, idmat_lr, var_names, shock_names, 40, 1000, ...
                 true, [], [], [], [], 16);   % 68% CI
close all;

disp('Output response to Supply shock at horizon 10:');
fprintf('90%% CI: [%.4f, %.4f]  Width: %.4f\n', ...
        EQ4a.Output_Supply(3,10), EQ4a.Output_Supply(1,10), ...
        EQ4a.Output_Supply(1,10) - EQ4a.Output_Supply(3,10));
fprintf('68%% CI: [%.4f, %.4f]  Width: %.4f\n', ...
        EQ4b.Output_Supply(3,10), EQ4b.Output_Supply(1,10), ...
        EQ4b.Output_Supply(1,10) - EQ4b.Output_Supply(3,10));
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 6: With seasonal dummies
%% ========================================================================
disp('========================================');
disp('EXAMPLE 6: GMM long-run SVAR with quarterly seasonal dummies');
disp('========================================');

EQ5 = gmmLRSVAR(data, 2, idmat_lr, var_names, shock_names, 40, 1000, ...
                true, [], 'quarter');
close all;

disp('Note: Seasonal dummies controlled for in estimation');
disp('Long-run effects computed net of seasonal patterns');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 7: With exogenous variables
%% ========================================================================
disp('========================================');
disp('EXAMPLE 7: GMM long-run SVAR with exogenous crisis dummy');
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

EQ6 = gmmLRSVAR(data_crisis, 2, idmat_lr, var_names, shock_names, 40, ...
                1000, true, [], [], exog);
close all;

disp('Exogenous variable coefficients estimated');
disp('Crisis dummy controlled for in GMM estimation');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 8: With restricted reduced-form VAR
%% ========================================================================
disp('========================================');
disp('EXAMPLE 8: GMM long-run SVAR with restricted reduced-form VAR');
disp('Restriction: Unemployment does not affect Output/Inflation in lags');
disp('========================================');

lr = [1 1 0;   % Output affected by output, inflation only
      1 1 0;   % Inflation affected by output, inflation only
      1 1 1];  % Unemployment affected by all variables

EQ7 = gmmLRSVAR(data, 2, idmat_lr, var_names, shock_names, 40, 1000, ...
                true, lr);
close all;

disp('Reduced-form restrictions on lags imposed via lr matrix');
disp('Long-run structural identification via GMM');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 9: Analyzing structural shocks
%% ========================================================================
disp('========================================');
disp('EXAMPLE 9: Structural shocks from GMM long-run estimation');
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

% Plot structural shocks
figure('Position', [100 100 1000 600]);
for i = 1:K
    subplot(K, 1, i);
    plot(shocks(i,:), 'k-', 'LineWidth', 1);
    title([shock_names{i} ' shock (GMM Long-Run)']);
    ylabel('Magnitude');
    grid on;
end
xlabel('Time');
sgtitle('Structural Shocks (GMM Long-Run Identification)');
fprintf('Structural shocks plot created (Figure 1)\n');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 10: Variance decomposition with long-run restrictions
%% ========================================================================
disp('========================================');
disp('EXAMPLE 10: Variance decomposition from GMM long-run SVAR');
disp('========================================');

% Extract variance decomposition at specific horizons
horizons = [1, 4, 12, 40];
disp('Variance decomposition of Output:');
disp('Horizon    Supply     Demand     Labor');
for h = horizons
    fprintf('%3d        %.2f%%      %.2f%%      %.2f%%\n', h, ...
            EQ1.vardecshock_Supply(1,h), ...
            EQ1.vardecshock_Demand(1,h), ...
            EQ1.vardecshock_Labor(1,h));
end
fprintf('\n');

disp('Note: At long horizons (h=40), only Supply shock contributes');
disp('to Output variance due to long-run restrictions.');
fprintf('\n');

% Plot variance decomposition
figure('Position', [100 100 1200 400]);
subplot(1, 3, 1);
plot(1:40, EQ1.vardecshock_Supply(1,:), 'k-', 'LineWidth', 2);
hold on;
plot(1:40, EQ1.vardecshock_Demand(1,:), 'k--', 'LineWidth', 2);
plot(1:40, EQ1.vardecshock_Labor(1,:), 'k:', 'LineWidth', 2);
title('Output Variance Decomposition');
xlabel('Horizon');
ylabel('Percentage');
legend('Supply', 'Demand', 'Labor', 'Location', 'best');
ylim([0 105]);
grid on;

subplot(1, 3, 2);
plot(1:40, EQ1.vardecshock_Supply(2,:), 'k-', 'LineWidth', 2);
hold on;
plot(1:40, EQ1.vardecshock_Demand(2,:), 'k--', 'LineWidth', 2);
plot(1:40, EQ1.vardecshock_Labor(2,:), 'k:', 'LineWidth', 2);
title('Inflation Variance Decomposition');
xlabel('Horizon');
ylabel('Percentage');
legend('Supply', 'Demand', 'Labor', 'Location', 'best');
ylim([0 105]);
grid on;

subplot(1, 3, 3);
plot(1:40, EQ1.vardecshock_Supply(3,:), 'k-', 'LineWidth', 2);
hold on;
plot(1:40, EQ1.vardecshock_Demand(3,:), 'k--', 'LineWidth', 2);
plot(1:40, EQ1.vardecshock_Labor(3,:), 'k:', 'LineWidth', 2);
title('Unemployment Variance Decomposition');
xlabel('Horizon');
ylabel('Percentage');
legend('Supply', 'Demand', 'Labor', 'Location', 'best');
ylim([0 105]);
grid on;

sgtitle('Variance Decomposition (GMM Long-Run SVAR)');
fprintf('Variance decomposition plot created (Figure 2)\n');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 11: GMM vs Blanchard-Quah (lrSVAR) comparison
%% ========================================================================
disp('========================================');
disp('EXAMPLE 11: Comparing GMM with Blanchard-Quah identification');
disp('For lower triangular restrictions, GMM should match lrSVAR');
disp('========================================');

% Estimate with Blanchard-Quah
EQ_bq = lrSVAR(data, 2, var_names, shock_names, 40, 1000, true);
close all;

disp('Impact matrices comparison:');
disp('GMM P:');
disp(EQ1.P);
disp('Blanchard-Quah P:');
disp(EQ_bq.P);

disp('Difference (should be small):');
disp(abs(EQ1.P - EQ_bq.P));
fprintf('Max absolute difference: %.6f\n', max(abs(EQ1.P(:) - EQ_bq.P(:))));
fprintf('\n');

% Compare IRFs
figure('Position', [100 100 1200 400]);
for i = 1:K
    subplot(1, 3, i);
    plot(1:40, EQ1.Output_Supply(2,:), 'k-', 'LineWidth', 2);
    hold on;
    plot(1:40, EQ_bq.Output_Supply(2,:), 'r--', 'LineWidth', 2);
    title([var_names{i} ' response to Supply shock']);
    xlabel('Horizon');
    ylabel('Response');
    legend('GMM', 'Blanchard-Quah', 'Location', 'best');
    grid on;
end
sgtitle('GMM vs Blanchard-Quah: IRF Comparison');
fprintf('GMM vs Blanchard-Quah comparison plot created (Figure 3)\n');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 12: Long-run effects visualization
%% ========================================================================
disp('========================================');
disp('EXAMPLE 12: Visualizing long-run vs short-run effects');
disp('========================================');

% Plot IRFs showing convergence to long-run effects
figure('Position', [100 100 1200 800]);

% Output responses to all shocks
subplot(3, 1, 1);
plot(1:40, EQ1.Output_Supply(2,:), 'k-', 'LineWidth', 2);
hold on;
plot(1:40, EQ1.Output_Demand(2,:), 'k--', 'LineWidth', 2);
plot(1:40, EQ1.Output_Labor(2,:), 'k:', 'LineWidth', 2);
plot(1:40, zeros(1,40), 'k-', 'LineWidth', 0.5);
% Mark long-run values
yline(sum(EQ1.Output_Supply(2,:)), 'r-', 'LineWidth', 1);
text(35, sum(EQ1.Output_Supply(2,:))+0.1, 'LR \neq 0', 'Color', 'r');
yline(0, 'b--', 'LineWidth', 1);
text(35, 0.3, 'LR = 0', 'Color', 'b');
title('Output Responses');
xlabel('Horizon');
ylabel('Response');
legend('Supply (LR \neq 0)', 'Demand (LR = 0)', 'Labor (LR = 0)', ...
       'Location', 'best');
grid on;

% Inflation responses to all shocks
subplot(3, 1, 2);
plot(1:40, EQ1.Inflation_Supply(2,:), 'k-', 'LineWidth', 2);
hold on;
plot(1:40, EQ1.Inflation_Demand(2,:), 'k--', 'LineWidth', 2);
plot(1:40, EQ1.Inflation_Labor(2,:), 'k:', 'LineWidth', 2);
plot(1:40, zeros(1,40), 'k-', 'LineWidth', 0.5);
yline(0, 'b--', 'LineWidth', 1);
text(35, 0.3, 'LR = 0', 'Color', 'b');
title('Inflation Responses');
xlabel('Horizon');
ylabel('Response');
legend('Supply (LR \neq 0)', 'Demand (LR \neq 0)', 'Labor (LR = 0)', ...
       'Location', 'best');
grid on;

% Unemployment responses to all shocks
subplot(3, 1, 3);
plot(1:40, EQ1.Unemployment_Supply(2,:), 'k-', 'LineWidth', 2);
hold on;
plot(1:40, EQ1.Unemployment_Demand(2,:), 'k--', 'LineWidth', 2);
plot(1:40, EQ1.Unemployment_Labor(2,:), 'k:', 'LineWidth', 2);
plot(1:40, zeros(1,40), 'k-', 'LineWidth', 0.5);
title('Unemployment Responses');
xlabel('Horizon');
ylabel('Response');
legend('Supply (LR \neq 0)', 'Demand (LR \neq 0)', 'Labor (LR \neq 0)', ...
       'Location', 'best');
grid on;

sgtitle('Long-Run Restrictions in Action');
fprintf('Long-run effects visualization plot created (Figure 4)\n');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 13: Overidentification test
%% ========================================================================
disp('========================================');
disp('EXAMPLE 13: Testing overidentifying restrictions');
disp('========================================');

% Just-identified model (exactly K(K-1)/2 = 3 zero restrictions)
idmat_just = [1  0  0;
              1  1  0;
              1  1  1];

% Overidentified model (4 zero restrictions = one extra)
idmat_over = [1  0  0;
              1  1  0;
              0  0  1];

EQ_just = gmmLRSVAR(data, 2, idmat_just, var_names, shock_names, 40, ...
                    1000, true);
EQ_over = gmmLRSVAR(data, 2, idmat_over, var_names, shock_names, 40, ...
                    1000, true);
close all;

disp('Just-identified model:');
fprintf('  Zero restrictions: %d (minimum required)\n', sum(idmat_just(:)==0));
disp('  Impact matrix P:');
disp(EQ_just.P);
fprintf('\n');

disp('Overidentified model:');
fprintf('  Zero restrictions: %d (one extra restriction)\n', sum(idmat_over(:)==0));
disp('  Impact matrix P:');
disp(EQ_over.P);
fprintf('\n');

disp('Note: GMM objective value can be used for overidentification test');
disp('(J-statistic has chi-squared distribution under null of valid restrictions)');
fprintf('Degrees of freedom for J-test: %d\n', sum(idmat_over(:)==0) - K*(K-1)/2);
fprintf('\n\n');

%% ========================================================================
disp('========================================');
disp('All examples completed successfully!');
disp('========================================');
disp('Key GMM Long-Run SVAR features demonstrated:');
disp('- GMM estimation with long-run zero restrictions');
disp('- Blanchard-Quah identification');
disp('- Overidentification (more restrictions than minimum)');
disp('- Verification of long-run restrictions via cumulative IRFs');
disp('- Custom shock sizes and confidence levels');
disp('- Seasonal dummies and exogenous variables');
disp('- Restricted reduced-form VAR');
disp('- Structural shock analysis and orthogonality');
disp('- Variance decomposition at long horizons');
disp('- Comparison with Blanchard-Quah (lrSVAR)');
disp('- Visualization of long-run vs short-run effects');
disp('- Overidentification testing framework');