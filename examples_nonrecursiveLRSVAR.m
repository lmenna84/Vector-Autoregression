%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% EXAMPLES FOR nonrecursiveLRSVAR
%% Author: Lorenzo Menna
%% This script demonstrates all features of the nonrecursiveLRSVAR function
%% (Non-recursive long-run SVAR identification)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all; clc;

%% ========================================================================
%% DATA SIMULATION
%% ========================================================================
% We simulate a 3-variable VAR(2) system for non-recursive long-run identification

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
disp('========================================');
fprintf('\n');

%% ========================================================================
%% EXAMPLE 1: Non-recursive long-run SVAR (symmetric pattern)
%% ========================================================================
disp('========================================');
disp('EXAMPLE 1: Non-recursive long-run SVAR with symmetric pattern');
disp('This pattern cannot be achieved with Blanchard-Quah');
disp('========================================');

var_names = {'Output', 'Inflation', 'Interest'};
shock_names = {'Productivity', 'Demand', 'Monetary'};

% Non-recursive pattern: symmetric zero restrictions
% Productivity affects Output and Inflation (not Interest) in LR
% Demand affects Inflation and Interest (not Output) in LR
% Monetary affects Output and Interest (not Inflation) in LR
idmat_nonrec = [1  0  1;    % Output affected by Productivity and Monetary
                1  1  0;    % Inflation affected by Productivity and Demand
                0  1  1];   % Interest affected by Demand and Monetary

% Check exact identification: must have K(K+1)/2 = 6 ones
fprintf('Number of ones in idmat: %d\n', sum(idmat_nonrec(:)));
fprintf('Required for exact identification: %d\n', K*(K+1)/2);

if sum(idmat_nonrec(:)) ~= K*(K+1)/2
    error('Identification matrix does not have exactly K(K+1)/2 ones!');
end

EQ1 = nonrecursiveLRSVAR(data, 2, idmat_nonrec, var_names, shock_names, ...
                         40, 1000, true);
close all;

disp('Estimated impact matrix P:');
disp(EQ1.P);
fprintf('\n');

disp('Long-run restriction pattern (zeros in long-run multiplier):');
disp('Productivity -> Interest: should be ~0');
disp('Demand -> Output: should be ~0');
disp('Monetary -> Inflation: should be ~0');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 2: Verifying non-recursive long-run restrictions
%% ========================================================================
disp('========================================');
disp('EXAMPLE 2: Verifying non-recursive long-run restrictions');
disp('========================================');

% Cumulative IRFs approximate long-run effects
cum_prod_interest = sum(EQ1.Interest_Productivity(2,:));
cum_demand_output = sum(EQ1.Output_Demand(2,:));
cum_monet_inflation = sum(EQ1.Inflation_Monetary(2,:));

disp('Cumulative IRFs (sum over 40 periods):');
fprintf('Productivity -> Interest: %.4f (should be ~0)\n', cum_prod_interest);
fprintf('Demand -> Output:         %.4f (should be ~0)\n', cum_demand_output);
fprintf('Monetary -> Inflation:    %.4f (should be ~0)\n', cum_monet_inflation);
fprintf('\n');

% Show allowed long-run effects
cum_prod_output = sum(EQ1.Output_Productivity(2,:));
cum_demand_inflation = sum(EQ1.Inflation_Demand(2,:));
cum_monet_interest = sum(EQ1.Interest_Monetary(2,:));

disp('Allowed long-run effects (non-zero):');
fprintf('Productivity -> Output:    %.4f\n', cum_prod_output);
fprintf('Demand -> Inflation:       %.4f\n', cum_demand_inflation);
fprintf('Monetary -> Interest:      %.4f\n', cum_monet_interest);
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 3: Comparison with recursive long-run SVAR
%% ========================================================================
disp('========================================');
disp('EXAMPLE 3: Non-recursive vs Recursive long-run identification');
disp('========================================');

% Recursive (Blanchard-Quah) identification
idmat_rec = [1  0  0;
             1  1  0;
             1  1  1];

EQ_rec = lrSVAR(data, 2, var_names, shock_names, 40, 1000, true);
close all;

disp('Non-recursive pattern:');
disp(idmat_nonrec);
disp('Allows symmetric restrictions impossible with recursive ordering');
fprintf('\n');

disp('Recursive (Blanchard-Quah) pattern:');
disp(idmat_rec);
disp('Restricted to lower triangular structure');
fprintf('\n');

disp('Impact matrix comparison:');
disp('Non-recursive P:');
disp(EQ1.P);
disp('Recursive P:');
disp(EQ_rec.P);
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 4: Different non-recursive patterns
%% ========================================================================
disp('========================================');
disp('EXAMPLE 4: Alternative non-recursive pattern');
disp('Each shock affects exactly 2 variables in the long run');
disp('========================================');

% Alternative symmetric pattern: each shock affects exactly 2 variables
% Supply affects Output and Employment (not Inflation) in LR
% Demand affects Output and Inflation (not Employment) in LR
% Monetary affects Employment and Inflation (not Output) in LR
var_names2 = {'Output', 'Employment', 'Inflation'};
shock_names2 = {'Supply', 'Demand', 'Monetary'};

idmat_alt = [1  1  0;    % Output: Supply ✓, Demand ✓, Monetary ✗
             1  0  1;    % Employment: Supply ✓, Demand ✗, Monetary ✓
             0  1  1];   % Inflation: Supply ✗, Demand ✓, Monetary ✓

% Verify exact identification
fprintf('Number of ones in idmat: %d\n', sum(idmat_alt(:)));
fprintf('Required for exact identification: %d\n', K*(K+1)/2);

if sum(idmat_alt(:)) ~= K*(K+1)/2
    error('Identification matrix does not have exactly K(K+1)/2 ones!');
end

EQ2 = nonrecursiveLRSVAR(data, 2, idmat_alt, var_names2, shock_names2, ...
                         40, 1000, true);
close all;

disp('Alternative symmetric pattern (each shock affects 2 variables):');
disp(idmat_alt);
fprintf('\n');

% Verify the restrictions
cum_supply_infl = sum(EQ2.Inflation_Supply(2,:));
cum_demand_empl = sum(EQ2.Employment_Demand(2,:));
cum_monet_output = sum(EQ2.Output_Monetary(2,:));

disp('Long-run restrictions verified:');
fprintf('Supply -> Inflation:  %.4f (should be ~0)\n', cum_supply_infl);
fprintf('Demand -> Employment: %.4f (should be ~0)\n', cum_demand_empl);
fprintf('Monetary -> Output:   %.4f (should be ~0)\n', cum_monet_output);
fprintf('\n');

disp('This pattern is symmetric and cannot be achieved with');
disp('recursive (Blanchard-Quah) identification.');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 5: Custom shock sizes
%% ========================================================================
disp('========================================');
disp('EXAMPLE 5: Non-recursive SVAR with custom shock sizes');
disp('========================================');

shock_size = 2 * diag(EQ1.P);  % 2 std deviations
EQ3 = nonrecursiveLRSVAR(data, 2, idmat_nonrec, var_names, shock_names, ...
                         40, 1000, true, [], [], [], shock_size);
close all;

disp('Comparison of IRF magnitudes at impact (horizon 2):');
disp('           1-SD shock    2-SD shock');
fprintf('Output:    %.4f        %.4f\n', ...
        EQ1.Output_Productivity(2,2), EQ3.Output_Productivity(2,2));
fprintf('Expected ratio: ~2.0, Actual ratio: %.4f\n', ...
        EQ3.Output_Productivity(2,2) / EQ1.Output_Productivity(2,2));

disp('Note: Index 1 is initial condition (zero), index 2 is impact effect');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 6: Different confidence levels
%% ========================================================================
disp('========================================');
disp('EXAMPLE 6: Non-recursive SVAR with different confidence intervals');
disp('========================================');

EQ4a = nonrecursiveLRSVAR(data, 2, idmat_nonrec, var_names, shock_names, ...
                          40, 1000, true, [], [], [], [], 5);    % 90% CI
EQ4b = nonrecursiveLRSVAR(data, 2, idmat_nonrec, var_names, shock_names, ...
                          40, 1000, true, [], [], [], [], 16);   % 68% CI
close all;

disp('Output response to Productivity shock at horizon 10 (index 10):');
fprintf('90%% CI: [%.4f, %.4f]  Width: %.4f\n', ...
        EQ4a.Output_Productivity(3,10), EQ4a.Output_Productivity(1,10), ...
        EQ4a.Output_Productivity(1,10) - EQ4a.Output_Productivity(3,10));
fprintf('68%% CI: [%.4f, %.4f]  Width: %.4f\n', ...
        EQ4b.Output_Productivity(3,10), EQ4b.Output_Productivity(1,10), ...
        EQ4b.Output_Productivity(1,10) - EQ4b.Output_Productivity(3,10));
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 7: With seasonal dummies
%% ========================================================================
disp('========================================');
disp('EXAMPLE 7: Non-recursive SVAR with quarterly seasonal dummies');
disp('========================================');

EQ5 = nonrecursiveLRSVAR(data, 2, idmat_nonrec, var_names, shock_names, ...
                         40, 1000, true, [], 'quarter');
close all;

disp('Seasonal dummies controlled for in estimation');
disp('Non-recursive long-run restrictions applied to deseasonalized data');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 8: With exogenous variables
%% ========================================================================
disp('========================================');
disp('EXAMPLE 8: Non-recursive SVAR with exogenous crisis dummy');
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

EQ6 = nonrecursiveLRSVAR(data_crisis, 2, idmat_nonrec, var_names, ...
                         shock_names, 40, 1000, true, [], [], exog);
close all;

disp('Crisis dummy controlled for in estimation');
disp('Non-recursive identification applied to crisis-adjusted data');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 9: With restricted reduced-form VAR
%% ========================================================================
disp('========================================');
disp('EXAMPLE 9: Non-recursive SVAR with restricted reduced-form VAR');
disp('========================================');

lr = [1 1 0;   % Output affected by output, inflation only
      1 1 0;   % Inflation affected by output, inflation only
      1 1 1];  % Interest affected by all variables

EQ7 = nonrecursiveLRSVAR(data, 2, idmat_nonrec, var_names, shock_names, ...
                         40, 1000, true, lr);
close all;

disp('Reduced-form restrictions on lags imposed via lr matrix');
disp('Non-recursive long-run structural identification applied');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 10: Structural shocks analysis
%% ========================================================================
disp('========================================');
disp('EXAMPLE 10: Structural shocks from non-recursive identification');
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
    title([shock_names{i} ' shock (Non-recursive LR)']);
    ylabel('Magnitude');
    grid on;
end
xlabel('Time');
sgtitle('Structural Shocks (Non-recursive Long-Run Identification)');
fprintf('Structural shocks plot created (Figure 1)\n');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 11: Variance decomposition with non-recursive restrictions
%% ========================================================================
disp('========================================');
disp('EXAMPLE 11: Variance decomposition analysis');
disp('========================================');

% Extract variance decomposition at specific horizons
horizons = [1, 4, 12, 40];

disp('Variance decomposition of Output:');
disp('Horizon    Productivity   Demand     Monetary');
for h = horizons
    fprintf('%3d        %.2f%%          %.2f%%     %.2f%%\n', h, ...
            EQ1.vardecshock_Productivity(1,h), ...
            EQ1.vardecshock_Demand(1,h), ...
            EQ1.vardecshock_Monetary(1,h));
end
fprintf('\n');

disp('Note: Demand has no long-run effect on Output');
fprintf('At h=40, Demand contribution: %.2f%%\n', ...
        EQ1.vardecshock_Demand(1,40));
fprintf('\n');

% Plot variance decomposition
figure('Position', [100 100 1200 400]);
subplot(1, 3, 1);
plot(1:40, EQ1.vardecshock_Productivity(1,:), 'k-', 'LineWidth', 2);
hold on;
plot(1:40, EQ1.vardecshock_Demand(1,:), 'k--', 'LineWidth', 2);
plot(1:40, EQ1.vardecshock_Monetary(1,:), 'k:', 'LineWidth', 2);
title('Output Variance Decomposition');
xlabel('Horizon');
ylabel('Percentage');
legend('Productivity', 'Demand', 'Monetary', 'Location', 'best');
ylim([0 105]);
grid on;

subplot(1, 3, 2);
plot(1:40, EQ1.vardecshock_Productivity(2,:), 'k-', 'LineWidth', 2);
hold on;
plot(1:40, EQ1.vardecshock_Demand(2,:), 'k--', 'LineWidth', 2);
plot(1:40, EQ1.vardecshock_Monetary(2,:), 'k:', 'LineWidth', 2);
title('Inflation Variance Decomposition');
xlabel('Horizon');
ylabel('Percentage');
legend('Productivity', 'Demand', 'Monetary', 'Location', 'best');
ylim([0 105]);
grid on;

subplot(1, 3, 3);
plot(1:40, EQ1.vardecshock_Productivity(3,:), 'k-', 'LineWidth', 2);
hold on;
plot(1:40, EQ1.vardecshock_Demand(3,:), 'k--', 'LineWidth', 2);
plot(1:40, EQ1.vardecshock_Monetary(3,:), 'k:', 'LineWidth', 2);
title('Interest Rate Variance Decomposition');
xlabel('Horizon');
ylabel('Percentage');
legend('Productivity', 'Demand', 'Monetary', 'Location', 'best');
ylim([0 105]);
grid on;

sgtitle('Variance Decomposition (Non-recursive LR SVAR)');
fprintf('Variance decomposition plot created (Figure 2)\n');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 12: Visualizing non-recursive long-run pattern
%% ========================================================================
disp('========================================');
disp('EXAMPLE 12: Visualizing the non-recursive restriction pattern');
disp('========================================');

% Create a visual representation of the restriction pattern
figure('Position', [100 100 800 600]);

% Original idmat (what affects what in long run)
subplot(1, 2, 1);
imagesc(idmat_nonrec);
colormap([1 1 1; 0 0 0]);  % White for 0, black for 1
set(gca, 'XTick', 1:K, 'XTickLabel', shock_names, ...
         'YTick', 1:K, 'YTickLabel', var_names);
xlabel('Shocks');
ylabel('Variables');
title('Long-Run Effects (idmat)');
for i = 1:K
    for j = 1:K
        text(j, i, num2str(idmat_nonrec(i,j)), ...
             'HorizontalAlignment', 'center', 'Color', 'r', 'FontSize', 14);
    end
end

% Zero pattern (what does NOT affect what)
subplot(1, 2, 2);
imagesc(1 - idmat_nonrec);
colormap([1 1 1; 0 0 0]);
set(gca, 'XTick', 1:K, 'XTickLabel', shock_names, ...
         'YTick', 1:K, 'YTickLabel', var_names);
xlabel('Shocks');
ylabel('Variables');
title('Zero Restrictions (LR)');
for i = 1:K
    for j = 1:K
        if idmat_nonrec(i,j) == 0
            text(j, i, 'X', 'HorizontalAlignment', 'center', ...
                 'Color', 'r', 'FontSize', 16, 'FontWeight', 'bold');
        end
    end
end

sgtitle('Non-Recursive Long-Run Restriction Pattern');
fprintf('Restriction pattern visualization created (Figure 3)\n');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 13: Long-run effects comparison
%% ========================================================================
disp('========================================');
disp('EXAMPLE 13: Comparing short-run vs long-run effects');
disp('Cumulative IRFs show long-run restrictions in action');
disp('========================================');

% Create figure with both IRFs and cumulative IRFs
figure('Position', [100 100 1400 800]);

% ===== OUTPUT RESPONSES =====
subplot(3, 2, 1);
plot(1:40, EQ1.Output_Productivity(2,:), 'k-', 'LineWidth', 2);
hold on;
plot(1:40, EQ1.Output_Demand(2,:), 'k--', 'LineWidth', 2);
plot(1:40, EQ1.Output_Monetary(2,:), 'k:', 'LineWidth', 2);
plot(1:40, zeros(1,40), 'k-', 'LineWidth', 0.5);
title('Output: IRFs');
xlabel('Horizon');
ylabel('Response');
legend('Productivity (LR \neq 0)', 'Demand (LR = 0)', 'Monetary (LR \neq 0)', ...
       'Location', 'best');
grid on;

subplot(3, 2, 2);
plot(1:40, cumsum(EQ1.Output_Productivity(2,:)), 'k-', 'LineWidth', 2);
hold on;
plot(1:40, cumsum(EQ1.Output_Demand(2,:)), 'k--', 'LineWidth', 2);
plot(1:40, cumsum(EQ1.Output_Monetary(2,:)), 'k:', 'LineWidth', 2);
plot(1:40, zeros(1,40), 'r--', 'LineWidth', 1.5);
title('Output: Cumulative IRFs (Long-Run Effects)');
xlabel('Horizon');
ylabel('Cumulative Response');
legend('Productivity (LR \neq 0)', 'Demand (LR = 0)', 'Monetary (LR \neq 0)', ...
       'Location', 'best');
grid on;
% Highlight which converge to zero
text(35, cumsum(EQ1.Output_Demand(2,end)), '\leftarrow LR = 0', ...
     'Color', 'b', 'FontSize', 10);

% ===== INFLATION RESPONSES =====
subplot(3, 2, 3);
plot(1:40, EQ1.Inflation_Productivity(2,:), 'k-', 'LineWidth', 2);
hold on;
plot(1:40, EQ1.Inflation_Demand(2,:), 'k--', 'LineWidth', 2);
plot(1:40, EQ1.Inflation_Monetary(2,:), 'k:', 'LineWidth', 2);
plot(1:40, zeros(1,40), 'k-', 'LineWidth', 0.5);
title('Inflation: IRFs');
xlabel('Horizon');
ylabel('Response');
legend('Productivity (LR \neq 0)', 'Demand (LR \neq 0)', 'Monetary (LR = 0)', ...
       'Location', 'best');
grid on;

subplot(3, 2, 4);
plot(1:40, cumsum(EQ1.Inflation_Productivity(2,:)), 'k-', 'LineWidth', 2);
hold on;
plot(1:40, cumsum(EQ1.Inflation_Demand(2,:)), 'k--', 'LineWidth', 2);
plot(1:40, cumsum(EQ1.Inflation_Monetary(2,:)), 'k:', 'LineWidth', 2);
plot(1:40, zeros(1,40), 'r--', 'LineWidth', 1.5);
title('Inflation: Cumulative IRFs (Long-Run Effects)');
xlabel('Horizon');
ylabel('Cumulative Response');
legend('Productivity (LR \neq 0)', 'Demand (LR \neq 0)', 'Monetary (LR = 0)', ...
       'Location', 'best');
grid on;
text(35, cumsum(EQ1.Inflation_Monetary(2,end)), '\leftarrow LR = 0', ...
     'Color', 'b', 'FontSize', 10);

% ===== INTEREST RATE RESPONSES =====
subplot(3, 2, 5);
plot(1:40, EQ1.Interest_Productivity(2,:), 'k-', 'LineWidth', 2);
hold on;
plot(1:40, EQ1.Interest_Demand(2,:), 'k--', 'LineWidth', 2);
plot(1:40, EQ1.Interest_Monetary(2,:), 'k:', 'LineWidth', 2);
plot(1:40, zeros(1,40), 'k-', 'LineWidth', 0.5);
title('Interest Rate: IRFs');
xlabel('Horizon');
ylabel('Response');
legend('Productivity (LR = 0)', 'Demand (LR \neq 0)', 'Monetary (LR \neq 0)', ...
       'Location', 'best');
grid on;

subplot(3, 2, 6);
plot(1:40, cumsum(EQ1.Interest_Productivity(2,:)), 'k-', 'LineWidth', 2);
hold on;
plot(1:40, cumsum(EQ1.Interest_Demand(2,:)), 'k--', 'LineWidth', 2);
plot(1:40, cumsum(EQ1.Interest_Monetary(2,:)), 'k:', 'LineWidth', 2);
plot(1:40, zeros(1,40), 'r--', 'LineWidth', 1.5);
title('Interest Rate: Cumulative IRFs (Long-Run Effects)');
xlabel('Horizon');
ylabel('Cumulative Response');
legend('Productivity (LR = 0)', 'Demand (LR \neq 0)', 'Monetary (LR \neq 0)', ...
       'Location', 'best');
grid on;
text(35, cumsum(EQ1.Interest_Productivity(2,end)), '\leftarrow LR = 0', ...
     'Color', 'b', 'FontSize', 10);

sgtitle('IRFs vs Cumulative IRFs: Non-Recursive Long-Run Restrictions');
fprintf('Long-run effects visualization created (Figure 4)\n');

% Print cumulative values to verify restrictions
fprintf('\n');
disp('Cumulative IRF values at horizon 40 (long-run effects):');
fprintf('Output:\n');
fprintf('  Productivity: %.4f (LR \neq 0)\n', sum(EQ1.Output_Productivity(2,:)));
fprintf('  Demand:       %.4f (LR = 0, should be ~0)\n', sum(EQ1.Output_Demand(2,:)));
fprintf('  Monetary:     %.4f (LR \neq 0)\n', sum(EQ1.Output_Monetary(2,:)));
fprintf('Inflation:\n');
fprintf('  Productivity: %.4f (LR \neq 0)\n', sum(EQ1.Inflation_Productivity(2,:)));
fprintf('  Demand:       %.4f (LR \neq 0)\n', sum(EQ1.Inflation_Demand(2,:)));
fprintf('  Monetary:     %.4f (LR = 0, should be ~0)\n', sum(EQ1.Inflation_Monetary(2,:)));
fprintf('Interest:\n');
fprintf('  Productivity: %.4f (LR = 0, should be ~0)\n', sum(EQ1.Interest_Productivity(2,:)));
fprintf('  Demand:       %.4f (LR \neq 0)\n', sum(EQ1.Interest_Demand(2,:)));
fprintf('  Monetary:     %.4f (LR \neq 0)\n', sum(EQ1.Interest_Monetary(2,:)));
fprintf('\n\n');

%% ========================================================================
disp('========================================');
disp('All examples completed successfully!');
disp('========================================');
disp('Key Non-recursive Long-Run SVAR features demonstrated:');
disp('- Non-recursive (non-triangular) long-run restriction patterns');
disp('- Symmetric zero restrictions impossible with Blanchard-Quah');
disp('- Block structure identification');
disp('- Exact identification requirement: K(K+1)/2 ones in idmat');
disp('- Verification of long-run restrictions via cumulative IRFs');
disp('- Custom shock sizes and confidence levels');
disp('- Seasonal dummies and exogenous variables');
disp('- Restricted reduced-form VAR');
disp('- Structural shock analysis and orthogonality');
disp('- Variance decomposition with non-recursive patterns');
disp('- Comparison with recursive long-run SVAR');
disp('- Visualization of restriction patterns');