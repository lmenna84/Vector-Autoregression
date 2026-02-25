%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% EXAMPLES FOR bloque_exo
%% Author: Lorenzo Menna
%% This script demonstrates all features of the bloque_exo function
%% (Block-exogenous SVAR with different sample sizes)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all; clc;

%% ========================================================================
%% DATA SIMULATION
%% ========================================================================
% We simulate a block-exogenous VAR system where:
% - Exogenous block: 2 variables with full sample (T1 = 250 observations)
% - Endogenous block: 2 variables with shorter sample (T2 = 150 observations)
% - Endogenous variables affected by exogenous, but not vice versa

rng(12345);  % Set seed for reproducibility

% Parameters
K1 = 2;       % Number of exogenous block variables (e.g., international)
K2 = 2;       % Number of endogenous block variables (e.g., domestic)
K = K1 + K2;  % Total variables
T1 = 250;     % Full sample for exogenous block
T2 = 150;     % Shorter sample for endogenous block
nlags = 2;    % Number of lags

% True VAR coefficient matrices (stable system)
% Block structure: exogenous block independent, endogenous affected by exogenous
A1_true = [0.5  0.1  0.0  0.0;    % Exog var 1: only affected by exog block
           0.1  0.6  0.0  0.0;    % Exog var 2: only affected by exog block
           0.2  0.1  0.5  0.1;    % Endo var 1: affected by both blocks
           0.1  0.2  0.1  0.4];   % Endo var 2: affected by both blocks

A2_true = [0.2  0.0  0.0  0.0;
           0.0  0.2  0.0  0.0;
           0.1  0.1  0.2  0.0;
           0.1  0.0  0.0  0.2];

nu = [0.5; 1.0; 0.3; 0.7];  % Constants

% True block-triangular impact matrix (Cholesky structure)
B_true = [1.0  0.0  0.0  0.0;    % Exogenous block
          0.4  1.2  0.0  0.0;    % (upper-left 2x2)
          0.3  0.2  0.8  0.0;    % Endogenous block affected by exogenous
          0.2  0.3  0.3  0.9];   % (lower-right 2x2 + lower-left linkage)

% Generate structural shocks (orthogonal)
epsilon = randn(K, T1);

% Generate VAR data for full sample
data_full = zeros(T1, K);
data_full(1:nlags, :) = randn(nlags, K);  % Initial values

for t = (nlags+1):T1
    structural_shock = B_true * epsilon(:, t);
    data_full(t, :) = nu' + data_full(t-1, :) * A1_true' + ...
                      data_full(t-2, :) * A2_true' + structural_shock';
end

% Split into exogenous block (full sample) and endogenous block (last T2 obs)
data_bloqueexo = data_full(:, 1:K1);                    % Full sample, exog vars
data_bloqueendo = data_full(end-T2+1:end, K1+1:K);      % Last T2 obs, endo vars

disp('========================================');
disp('Data simulation complete');
disp(['Exogenous block: ' num2str(K1) ' variables, ' num2str(T1) ' observations']);
disp(['Endogenous block: ' num2str(K2) ' variables, ' num2str(T2) ' observations']);
disp('Block-exogenous structure: exog → endo, but not endo → exog');
disp('========================================');
fprintf('\n');

%% ========================================================================
%% EXAMPLE 1: Basic block-exogenous SVAR
%% ========================================================================
disp('========================================');
disp('EXAMPLE 1: Basic block-exogenous SVAR');
disp('International variables (exog) vs Domestic variables (endo)');
disp('========================================');

var_names = {'WorldGDP', 'WorldInterest', 'DomesticGDP', 'DomesticInterest'};
shock_names = {'WorldSupply', 'WorldMonetary', 'DomesticSupply', 'DomesticMonetary'};

EQ1 = bloque_exo(data_bloqueexo, data_bloqueendo, nlags, var_names, shock_names);
close all;

disp('Block-triangular impact matrix P:');
disp(EQ1.P);
fprintf('\n');

disp('Key features:');
disp('  - Upper-left 2x2 block: exogenous block identification (full sample)');
disp('  - Lower-right 2x2 block: endogenous block conditional identification');
disp('  - Upper-right zeros: endogenous shocks do not affect exogenous vars');
disp('  - Lower-left non-zeros: exogenous shocks affect endogenous vars');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 2: Examining the block structure
%% ========================================================================
disp('========================================');
disp('EXAMPLE 2: Understanding the block-exogenous structure');
disp('========================================');

% Extract blocks from P matrix
P_exo = EQ1.P(1:K1, 1:K1);              % Upper-left: exogenous identification
P_link = EQ1.P(K1+1:K, 1:K1);           % Lower-left: linkage from exog to endo
P_endo = EQ1.P(K1+1:K, K1+1:K);         % Lower-right: endogenous identification

disp('Exogenous block impact matrix (2x2):');
disp(P_exo);
fprintf('\n');

disp('Linkage matrix (endogenous on exogenous shocks):');
disp(P_link);
fprintf('\n');

disp('Endogenous block impact matrix (2x2):');
disp(P_endo);
fprintf('\n');

disp('Interpretation:');
disp('  - World shocks identified using FULL sample (250 obs)');
disp('  - Domestic shocks identified using OVERLAPPING sample (150 obs)');
disp('  - Domestic variables respond to world shocks contemporaneously');
disp('  - World variables DO NOT respond to domestic shocks');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 3: Structural shocks for overlapping sample
%% ========================================================================
disp('========================================');
disp('EXAMPLE 3: Structural shocks for overlapping period');
disp('========================================');

T_effective = T2 - nlags;  % Effective sample after losing observations to lags

disp('EQ.struc contains structural shocks for the overlapping T2 period');
fprintf('Size of structural shocks matrix: %dx%d\n', size(EQ1.struc, 1), size(EQ1.struc, 2));
fprintf('(%d shocks × %d effective observations after %d lags)\n', K, T_effective, nlags);
fprintf('\n');

% Plot structural shocks
figure('Position', [100 100 1200 800]);
for i = 1:K
    subplot(K, 1, i);
    plot(1:T_effective, EQ1.struc(i,:), 'k-', 'LineWidth', 1);
    hold on;
    plot(1:T_effective, zeros(1,T_effective), 'r--', 'LineWidth', 0.5);
    title([shock_names{i} ' (structural shock)']);
    xlabel('Time (effective sample)');
    ylabel('Shock');
    grid on;
end
sgtitle('Structural Shocks (Overlapping Sample Period)');
fprintf('Structural shocks plot created (Figure 1)\n');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 4: Custom shock sizes
%% ========================================================================
disp('========================================');
disp('EXAMPLE 4: Block-exogenous SVAR with custom shock sizes');
disp('========================================');

shock_size = 2 * diag(EQ1.P);  % 2 std deviations

% Set same random seed for fair comparison
rng(12345);
EQ1_reseed = bloque_exo(data_bloqueexo, data_bloqueendo, nlags, var_names, ...
                        shock_names, 40, 1000, true);

rng(12345);  % Same seed
EQ2 = bloque_exo(data_bloqueexo, data_bloqueendo, nlags, var_names, ...
                 shock_names, 40, 1000, true, [], [], [], [], shock_size);
close all;

disp('Comparison at horizon 10 (using same Monte Carlo draws):');
fprintf('WorldSupply → DomesticGDP (1-SD):  %.4f\n', EQ1_reseed.DomesticGDP_WorldSupply(2,10));
fprintf('WorldSupply → DomesticGDP (2-SD):  %.4f\n', EQ2.DomesticGDP_WorldSupply(2,10));
fprintf('Ratio: %.4f (expected: ~2.0)\n', ...
        EQ2.DomesticGDP_WorldSupply(2,10) / EQ1_reseed.DomesticGDP_WorldSupply(2,10));
fprintf('\n');

disp('Custom shock sizes successfully applied');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 5: Different confidence levels
%% ========================================================================
disp('========================================');
disp('EXAMPLE 5: Block-exogenous SVAR with different confidence intervals');
disp('========================================');

EQ3a = bloque_exo(data_bloqueexo, data_bloqueendo, nlags, var_names, ...
                  shock_names, 40, 1000, true, [], [], [], [], [], 5);    % 90% CI
EQ3b = bloque_exo(data_bloqueexo, data_bloqueendo, nlags, var_names, ...
                  shock_names, 40, 1000, true, [], [], [], [], [], 16);   % 68% CI
close all;

disp('DomesticGDP response to WorldSupply shock at horizon 10:');
fprintf('90%% CI: [%.4f, %.4f]  Width: %.4f\n', ...
        EQ3a.DomesticGDP_WorldSupply(3,10), EQ3a.DomesticGDP_WorldSupply(1,10), ...
        EQ3a.DomesticGDP_WorldSupply(1,10) - EQ3a.DomesticGDP_WorldSupply(3,10));
fprintf('68%% CI: [%.4f, %.4f]  Width: %.4f\n', ...
        EQ3b.DomesticGDP_WorldSupply(3,10), EQ3b.DomesticGDP_WorldSupply(1,10), ...
        EQ3b.DomesticGDP_WorldSupply(1,10) - EQ3b.DomesticGDP_WorldSupply(3,10));
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 6: With seasonal dummies
%% ========================================================================
disp('========================================');
disp('EXAMPLE 6: Block-exogenous SVAR with quarterly seasonal dummies');
disp('========================================');

EQ4 = bloque_exo(data_bloqueexo, data_bloqueendo, nlags, var_names, ...
                 shock_names, 40, 1000, true, [], [], 'quarter');
close all;

disp('Seasonal patterns controlled for in both blocks');
disp('Dummy coefficients stored in EQ.dummies');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 7: With exogenous variables
%% ========================================================================
disp('========================================');
disp('EXAMPLE 7: Block-exogenous SVAR with exogenous oil price shock');
disp('========================================');

% Create oil price shock dummy (affects full sample)
exog = zeros(T1, 1);
exog(120:125) = 1;  % Oil shock period

% Add oil shock effect to data
data_bloqueexo_oil = data_bloqueexo;
data_bloqueendo_oil = data_bloqueendo;
oil_effect_exo = [0.3; 0.5];  % Affects world variables
oil_effect_endo = [0.2; 0.4]; % Affects domestic variables
for t = 120:125
    data_bloqueexo_oil(t, :) = data_bloqueexo_oil(t, :) + oil_effect_exo';
end
for t = max(1, 120-(T1-T2)):min(T2, 125-(T1-T2))  % Adjust for different sample
    data_bloqueendo_oil(t, :) = data_bloqueendo_oil(t, :) + oil_effect_endo';
end

EQ5 = bloque_exo(data_bloqueexo_oil, data_bloqueendo_oil, nlags, var_names, ...
                 shock_names, 40, 1000, true, [], [], [], exog);
close all;

disp('Oil price shock controlled for via exogenous dummy');
disp('Exogenous variable coefficients stored in EQ.exo');
fprintf('\n\n');


%% ========================================================================
%% EXAMPLE 8: Variance decomposition analysis
%% ========================================================================
disp('========================================');
disp('EXAMPLE 8: Variance decomposition in block-exogenous system');
disp('========================================');

disp('Variance decomposition of DomesticGDP:');
disp('Horizon    WorldSupply  WorldMon  DomSupply  DomMon');
for h = [1, 4, 12, 20, 40]
    fprintf('%3d        %.2f%%       %.2f%%    %.2f%%     %.2f%%\n', h, ...
            EQ1.vardecshock_WorldSupply(3,h), ...
            EQ1.vardecshock_WorldMonetary(3,h), ...
            EQ1.vardecshock_DomesticSupply(3,h), ...
            EQ1.vardecshock_DomesticMonetary(3,h));
end
fprintf('\n');

% Plot variance decomposition
figure('Position', [100 100 1200 800]);

% Domestic GDP variance decomposition
subplot(2, 2, 1);
plot(1:40, EQ1.vardecshock_WorldSupply(3,:), 'k-', 'LineWidth', 2);
hold on;
plot(1:40, EQ1.vardecshock_WorldMonetary(3,:), 'k--', 'LineWidth', 2);
plot(1:40, EQ1.vardecshock_DomesticSupply(3,:), 'k:', 'LineWidth', 2);
plot(1:40, EQ1.vardecshock_DomesticMonetary(3,:), 'k-.', 'LineWidth', 2);
title('Domestic GDP Variance Decomposition');
xlabel('Horizon');
ylabel('Percentage');
legend('World Supply', 'World Monetary', 'Domestic Supply', 'Domestic Monetary', ...
       'Location', 'best');
ylim([0 105]);
grid on;

% Domestic Interest variance decomposition
subplot(2, 2, 2);
plot(1:40, EQ1.vardecshock_WorldSupply(4,:), 'k-', 'LineWidth', 2);
hold on;
plot(1:40, EQ1.vardecshock_WorldMonetary(4,:), 'k--', 'LineWidth', 2);
plot(1:40, EQ1.vardecshock_DomesticSupply(4,:), 'k:', 'LineWidth', 2);
plot(1:40, EQ1.vardecshock_DomesticMonetary(4,:), 'k-.', 'LineWidth', 2);
title('Domestic Interest Variance Decomposition');
xlabel('Horizon');
ylabel('Percentage');
legend('World Supply', 'World Monetary', 'Domestic Supply', 'Domestic Monetary', ...
       'Location', 'best');
ylim([0 105]);
grid on;

% World GDP variance decomposition
subplot(2, 2, 3);
plot(1:40, EQ1.vardecshock_WorldSupply(1,:), 'k-', 'LineWidth', 2);
hold on;
plot(1:40, EQ1.vardecshock_WorldMonetary(1,:), 'k--', 'LineWidth', 2);
plot(1:40, EQ1.vardecshock_DomesticSupply(1,:), 'k:', 'LineWidth', 2);
plot(1:40, EQ1.vardecshock_DomesticMonetary(1,:), 'k-.', 'LineWidth', 2);
title('World GDP Variance Decomposition');
xlabel('Horizon');
ylabel('Percentage');
legend('World Supply', 'World Monetary', 'Domestic Supply', 'Domestic Monetary', ...
       'Location', 'best');
ylim([0 105]);
grid on;

% World Interest variance decomposition
subplot(2, 2, 4);
plot(1:40, EQ1.vardecshock_WorldSupply(2,:), 'k-', 'LineWidth', 2);
hold on;
plot(1:40, EQ1.vardecshock_WorldMonetary(2,:), 'k--', 'LineWidth', 2);
plot(1:40, EQ1.vardecshock_DomesticSupply(2,:), 'k:', 'LineWidth', 2);
plot(1:40, EQ1.vardecshock_DomesticMonetary(2,:), 'k-.', 'LineWidth', 2);
title('World Interest Variance Decomposition');
xlabel('Horizon');
ylabel('Percentage');
legend('World Supply', 'World Monetary', 'Domestic Supply', 'Domestic Monetary', ...
       'Location', 'best');
ylim([0 105]);
grid on;

sgtitle('Variance Decomposition (Block-Exogenous SVAR)');
fprintf('Variance decomposition plot created (Figure 2)\n');
fprintf('\n');

disp('Note: World variables NOT affected by domestic shocks (block structure)');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 9: Spillover from world to domestic
%% ========================================================================
disp('========================================');
disp('EXAMPLE 9: Analyzing spillovers from world to domestic economy');
disp('========================================');

% Plot spillover IRFs
figure('Position', [100 100 1200 600]);

% World supply shock spillover
subplot(2, 2, 1);
x_fill = [1:40, fliplr(1:40)];
y_fill = [EQ1.DomesticGDP_WorldSupply(1,:), fliplr(EQ1.DomesticGDP_WorldSupply(3,:))];
fill(x_fill, y_fill, [0.8 0.8 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
hold on;
plot(1:40, EQ1.DomesticGDP_WorldSupply(2,:), 'k-', 'LineWidth', 2);
plot(1:40, zeros(1,40), 'r--', 'LineWidth', 0.5);
title('Domestic GDP ← World Supply Shock');
xlabel('Horizon');
ylabel('Response');
grid on;

% World monetary shock spillover
subplot(2, 2, 2);
y_fill = [EQ1.DomesticGDP_WorldMonetary(1,:), fliplr(EQ1.DomesticGDP_WorldMonetary(3,:))];
fill(x_fill, y_fill, [0.8 0.8 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
hold on;
plot(1:40, EQ1.DomesticGDP_WorldMonetary(2,:), 'k-', 'LineWidth', 2);
plot(1:40, zeros(1,40), 'r--', 'LineWidth', 0.5);
title('Domestic GDP ← World Monetary Shock');
xlabel('Horizon');
ylabel('Response');
grid on;

% World supply shock spillover to interest
subplot(2, 2, 3);
y_fill = [EQ1.DomesticInterest_WorldSupply(1,:), fliplr(EQ1.DomesticInterest_WorldSupply(3,:))];
fill(x_fill, y_fill, [0.8 0.8 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
hold on;
plot(1:40, EQ1.DomesticInterest_WorldSupply(2,:), 'k-', 'LineWidth', 2);
plot(1:40, zeros(1,40), 'r--', 'LineWidth', 0.5);
title('Domestic Interest ← World Supply Shock');
xlabel('Horizon');
ylabel('Response');
grid on;

% World monetary shock spillover to interest
subplot(2, 2, 4);
y_fill = [EQ1.DomesticInterest_WorldMonetary(1,:), fliplr(EQ1.DomesticInterest_WorldMonetary(3,:))];
fill(x_fill, y_fill, [0.8 0.8 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
hold on;
plot(1:40, EQ1.DomesticInterest_WorldMonetary(2,:), 'k-', 'LineWidth', 2);
plot(1:40, zeros(1,40), 'r--', 'LineWidth', 0.5);
title('Domestic Interest ← World Monetary Shock');
xlabel('Horizon');
ylabel('Response');
grid on;

sgtitle('International Spillovers to Domestic Economy');
fprintf('Spillover analysis plot created (Figure 3)\n');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 10: Standard deviations of IRFs
%% ========================================================================
disp('========================================');
disp('EXAMPLE 10: Uncertainty in IRF estimates (EQ.stdirf)');
disp('========================================');

disp('Standard deviation of DomesticGDP response to WorldSupply at h=10:');
fprintf('  Point estimate: %.4f\n', EQ1.DomesticGDP_WorldSupply(2,10));
fprintf('  Std deviation:  %.4f\n', EQ1.stdirf(3,10,1));
fprintf('  90%% CI:         [%.4f, %.4f]\n', ...
        EQ1.DomesticGDP_WorldSupply(3,10), EQ1.DomesticGDP_WorldSupply(1,10));
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 11: Verbose option
%% ========================================================================
disp('========================================');
disp('EXAMPLE 11: Using verbose parameter');
disp('With verbose=true, Monte Carlo iteration numbers are printed');
disp('========================================');

disp('Running with verbose=true (will print iterations)...');
EQ7 = bloque_exo(data_bloqueexo, data_bloqueendo, nlags, var_names, ...
                 shock_names, 40, 100, true, [], [], [], [], [], 5, true);
close all;

disp(' ');
disp('With verbose=false (default), no iteration numbers printed');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 12: When to use block-exogenous approach
%% ========================================================================
disp('========================================');
disp('EXAMPLE 12: When to use block-exogenous SVAR');
disp('========================================');

disp('Use block-exogenous SVAR when:');
disp('  1. You have variables that are plausibly exogenous to others');
disp('     (e.g., world variables vs small open economy domestic variables)');
disp('  ');
disp('  2. Data availability differs across variable groups');
disp('     (e.g., international data available for longer period)');
disp('  ');
disp('  3. You want to exploit full sample for exogenous block identification');
disp('     (more efficient use of available data)');
disp('  ');
disp('  4. Economic theory supports block-exogeneity');
disp('     (e.g., large country vs small country, aggregate vs sectoral)');
fprintf('\n');

disp('Key advantages:');
disp('  - More efficient identification of exogenous block (uses full sample)');
disp('  - Imposes economically motivated restrictions (exogeneity)');
disp('  - Handles unbalanced sample sizes naturally');
fprintf('\n');

disp('Applications:');
disp('  - Small open economy models (world → domestic)');
disp('  - Sectoral analysis (aggregate → sector-specific)');
disp('  - Regional analysis (national → regional)');
disp('  - Financial spillovers (international → domestic financial markets)');
fprintf('\n\n');

%% ========================================================================
disp('========================================');
disp('All examples completed successfully!');
disp('========================================');
disp('Key Block-Exogenous SVAR features demonstrated:');
disp('- Basic block-exogenous structure');
disp('- Block-triangular impact matrix P');
disp('- Structural shocks for overlapping sample');
disp('- Custom shock sizes and confidence levels');
disp('- Seasonal dummies and exogenous variables');
disp('- Restricted reduced-form VAR (lrexo and lrendo)');
disp('- Variance decomposition');
disp('- International spillover analysis');
disp('- Standard deviations of IRFs (EQ.stdirf)');
disp('- Verbose option for debugging');
disp('- When to use block-exogenous approach');
disp(' ');
disp('Key concepts:');
disp('- Exogenous block identified using FULL sample');
disp('- Endogenous block identified using OVERLAPPING sample');
disp('- Block-triangular structure: exog → endo, but not endo → exog');
disp('- More efficient use of data when samples differ');
disp('- Economically motivated restrictions via block structure');