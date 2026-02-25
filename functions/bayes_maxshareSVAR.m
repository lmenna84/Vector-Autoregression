function EQ=bayes_maxshareSVAR(data,H,nlags,accum,var_names,shock_names,irf_lenght,gibbs,const,lr,dum,exog,shock,confianza,parbayes,burnin,verbose)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Author: Lorenzo Menna %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Purpose: 
% Impulse responses and Variance decomposition of a Bayesian Structural Vector Autoregression
% with max share identification (Uhlig, 2003; Barsky and Sims, 2011), with optional 
% restrictions on variables' interactions and Minnesota Independent Normal Wishart prior.
% The prior on the error variance matrix is uninformative.
%
% CRITICAL: The first shock is identified as the shock that maximizes the variance
% contribution to the FIRST VARIABLE at horizon H. The order of variables in the
% data matrix matters! Place the variable of interest (whose variance you want to
% maximize) as the FIRST column of the data matrix.
% -----------------------------------
% Inputs:
% data = NxK matrix (N number of observations, K number of variables).
% **IMPORTANT**: The FIRST variable is whose variance is maximized.
% H = scalar. Horizon at which to maximize the variance decomposition 
% nlags = desired number of lags (no maximum limit enforced)
% Optional:
% accum = scalar, either 1 or 0. 1 (default) if one wants to maximize the
%         cumulative variance decomposition across all horizons from 0 to H, 
%         0 if it is maximized only at the specified horizon. 
%         Note: Reported variance decomposition is always cumulative. With accum=0,
%         the first shock explains 100% of non-cumulative variance at H in each draw,
%         but the median across draws may differ from 100% in cumulative variance.
% var_names = 1xK cell vector with names of the variables
% shock_names = 1xK cell vector with names of the shocks. Default is same as var_names.
%               The first shock is typically given an economic interpretation (e.g., 'TFP shock')
% irf_lenght = desired length of the IRFs. Default is 40.
% gibbs = number of Gibbs sampler repetitions for the computation of
%         the confidence intervals of the IRFs. Default is 2000.
% const = true if the VAR is estimated with a constant and false if without.
%         Default is false.
% lr = KxK matrix of zeros and ones, zeros for variables that do not have
%      effects on other variables, one otherwise. Default is no restrictions.
% dum = seasonal dummies. Specify 'month' for 11 monthly dummies or 
%       'quarter' for 3 quarterly dummies. Default is no dummies.
% exog = NxE matrix, where E is the number of exogenous variables that
%        enter without lags (non-dynamic regressors, e.g., event dummies)
% shock = Kx1 vector containing the size of the shock for each variable. 
%         Default is one standard deviation shocks.
% confianza = level in % of the lowest confidence band. Default is 16%.
% parbayes = 1x6 vector. The first entry is the prior mean on the first own
%            lag autoregressive coefficient (default = 1), second entry is its prior
%            variance (default = 0.2), third entry is the ratio of prior variance on
%            cross-lags and own lags, before adjusting for relative standard deviation
%            (default = 1), fourth entry is the decay parameter on lags (default = 2),
%            fifth entry is the prior variance of the constant (default = 10^2), sixth
%            entry is the prior variance on exogenous variables (default = 10^2).
% burnin = number of burn-in iterations to discard. Default is 500.
% verbose = true to print iteration numbers, false to suppress. Default is false.
% -----------------------------------
% Returns:
% EQ = structure with the following fields:
% EQ.vari_shockj = K*K collections of 3xirf_lenght matrices containing in the first row the
%                  84th percentile, in the second row the median IRF, and in the third row the
%                  16th percentile. i is the variable, j is the shock.
%                  **Note**: Typically only the first shock (max share shock) is analyzed.
% EQ.B = KxK posterior mean of the identification matrix
% EQ.vardecshock_shockj = K matrices of size Kxirf_lenght containing in each row j the
%                         percentage of variance of variable j due to shock i. Each column
%                         is the number of periods ahead.
% EQ.constant_mean = Kx1 posterior mean of the constant (if const=true)
% EQ.exo_mean = KxE posterior mean of the coefficients on the exogenous variables (if exog provided)
% EQ.coefficient_mean = Kx(K*nlags) posterior mean of the VAR coefficients
% EQ.coeff_sim = Kx(total_coeffs)xgibbs array of all coefficient draws
% EQ.irf_sim = Kxirf_lenghtxKxgibbs array of all IRF draws
% ------------------------------------
% The solution method follows Lütkepohl (2005) and Koop & Korobilis (2010)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% beginnings...
%%%%%%%%%%%%%%%%%%%%%%
if nargin<17
    verbose=false;
end
if nargin<16
    verbose=false;
    burnin=500;
end
if nargin<15
    verbose=false;
    burnin=500;
    parbayes=[];
end
if nargin<14
    verbose=false;
    burnin=500;
    parbayes=[];
    confianza=16;
end
if nargin<13
    verbose=false;
    burnin=500;
    parbayes=[];
    confianza=16;
    shock=[];
end
if nargin<12
    verbose=false;
    burnin=500;
    parbayes=[];
    confianza=16;
    shock=[];
    exog=[];
end
if nargin<11
    verbose=false;
    burnin=500;
    parbayes=[];
    confianza=16;
    shock=[];
    exog=[];
    dum=[];
end
if nargin<10
    verbose=false;
    burnin=500;
    parbayes=[];
    confianza=16; 
    shock=[];
    exog=[];
    dum=[];
    lr=[];
end
if nargin<9
    verbose=false;
    burnin=500;
    parbayes=[];
    confianza=16; 
    shock=[];
    exog=[];
    dum=[];
    const=false;
    lr=[];
end
if nargin<8
    verbose=false;
    burnin=500;
    parbayes=[];
    confianza=16;   
    shock=[];
    exog=[];
    dum=[];
    const=false;
    lr=[];
    gibbs=2000;
end
if nargin<7
    verbose=false;
    burnin=500;
    parbayes=[];
    confianza=16;    
    shock=[];
    exog=[];
    dum=[];
    const=false;
    lr=[];
    gibbs=2000;
    irf_lenght=40;
end
if nargin<6
    verbose=false;
    burnin=500;
    parbayes=[];
    confianza=16;   
    shock=[];
    exog=[];
    dum=[];
    const=false;
    lr=[];
    gibbs=2000;
    irf_lenght=40;
    for xx=1:size(data,2)
        eval(['shock_names{' int2str(xx) '}=''var' int2str(xx) ''';']);
    end
end
if nargin<5
    verbose=false;
    burnin=500;
    parbayes=[];
    confianza=16;   
    shock=[];
    exog=[];
    dum=[];
    const=false;
    lr=[];
    gibbs=2000;
    irf_lenght=40;
    for xx=1:size(data,2)
        eval(['shock_names{' int2str(xx) '}=''var' int2str(xx) ''';']);
    end
    for xx=1:size(data,2)
        eval(['var_names{' int2str(xx) '}=''var' int2str(xx) ''';']);
    end
end
if nargin<4
    verbose=false;
    burnin=500;
    parbayes=[];
    confianza=16;   
    shock=[];
    exog=[];
    dum=[];
    const=false;
    lr=[];
    gibbs=2000;
    irf_lenght=40;
    for xx=1:size(data,2)
        eval(['shock_names{' int2str(xx) '}=''var' int2str(xx) ''';']);
    end
    for xx=1:size(data,2)
        eval(['var_names{' int2str(xx) '}=''var' int2str(xx) ''';']);
    end
    accum=1;
end

if isempty(verbose)==1
    verbose=false;
end
if isempty(burnin)==1
    burnin=500;
end
if isempty(parbayes)==1
    alf1=1;
    alf2=0.2;
    alf3=1;
    decay=2;
    alf4=10^2;
    alf5=10^2;
else alf1=parbayes(1);
    alf2=parbayes(2);
    alf3=parbayes(3);
    decay=parbayes(4);
    alf4=parbayes(5);
    alf5=parbayes(6);
end
if isempty(shock)==1
    shock=[];
end
if isempty(gibbs)==1
    gibbs=2000;
end
if isempty(irf_lenght)==1
    irf_lenght=40;
end
if isempty(confianza)==1
    confianza=16;
end
if isempty(shock_names)==1
    for xx=1:size(data,2)
        eval(['shock_names{' int2str(xx) '}=''var' int2str(xx) ''';']);
    end
end
if isempty(var_names)==1
    for xx=1:size(data,2)
        eval(['var_names{' int2str(xx) '}=''var' int2str(xx) ''';']);
    end
end
if isempty(accum)==1
    accum=1;
end
if isempty(dum)==1
    numdum=0;
elseif strcmpi('month',dum)==1
    numdum=11;
elseif strcmpi('quarter',dum)==1
    numdum=3;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Start with OLS estimation of the VAR and also obtain the Cholesky
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

K=size(data,2);
T=size(data,1)-nlags;
Q=reducedformVAR(data,nlags,const,lr,[],dum,exog);
% Put all coefficients in a single matrix: this is needed later for the
% Bayesian part
demm=isfield(Q,'constants');
demmexo=isfield(Q,'exo');
if demm==1 & demmexo==1
    coeff=[Q.constants Q.exo Q.coefficients];
elseif demm==0 & demmexo==0
    coeff=[Q.coefficients];
elseif demm==1 & demmexo==0
    coeff=[Q.constants Q.coefficients];
end
V=zeros(K*nlags,1+numdum);
demm=isfield(Q,'constants');
if demm==1
    V(1:K,1)=Q.constants;
end
Vexo=zeros(K*nlags,size(exog,2));
if isempty(exog)==0
    Vexo(1:K,:)=Q.exo;
end
if isempty(dum)==0
    V(1:K,2:numdum+1)=Q.dummies;
end
A=zeros(K*nlags,K*nlags);
A(1:K,:)=Q.coefficients;
J_mat=zeros(K,K*nlags);
J_mat(:,1:K)=eye(K);
if nlags>1
    A(K+1:K*nlags,1:K*nlags-K)=eye(K*nlags-K);
end

% Also build matrix of independent variables and dependent for use in the Bayesian part
Zprep=flipud(data(1:size(data,1)-1,:))';
Zprep=reshape(Zprep,(size(data,1)-1)*K,1);
if demm==1
    Zprep1=ones(1,T);
else Zprep1=[];
end
Zprepexo=exog(nlags+1:end,:)';
Zprep2=zeros(0,0);
count=0;
for xx=1:T
    Zprep2=[Zprep2 Zprep(1+count:nlags*K+count,1)];
    count=count+K;
end
Zprep2=fliplr(Zprep2);
Z=[Zprep1;Zprepexo;Zprep2];
Y=data(nlags+1:size(data,1),:)';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Now, set the hyperparameters of the prior distributions. 
%%%%%%%%%%%%%%%%%%%

% Minnesota prior: we set to alf1 own variable lags and to zero all other
% coefficients as prior means
coeff_prior=zeros(size(coeff));
if demm==1 & demmexo==1
    coeff_prior(:,1+size(Q.exo,2)+1:1+size(Q.exo,2)+K)=alf1*eye(K);
elseif demm==0 & demmexo==0
    coeff_prior(:,1:K)=alf1*eye(K);
elseif demm==1 & demmexo==0
    coeff_prior(:,2:1+K)=alf1*eye(K);
end
veccoeff_prior=coeff_prior(:);
% Now, I want to set the variance of the prior distribution of the
% coefficients. Start with the hyperparameters:
par_ownlag = alf2;
par_crosslag = alf3*par_ownlag;
par_decay = decay;
if demm==1
    par_const = alf4;
end
if demmexo==1
    par_exog = alf5;
end
% Now, compute the residual variances of univariate p-lag
% autoregressions. Notice that both the constant and the exogenous
% variables are also disregarded
for xx=1:K
    RF=reducedformVAR(data(:,xx),nlags,0);
    sig(xx,1)=RF.sigma;
end
% index position of the own lags in the autoregressive structure
ind=eye(K);
% Now, initialize a matrix in which I will put the variance of the
% prior distribution of the coefficient, but without including the
% covariances between the coefficients; so it will have the size of
% coeff
% First set the variance of the prior distribution of the coefficient
% on the constant and exogenous variables if they exist
if demm==1 & demmexo==1 
    V_i(:,1:1+size(Q.exo,2))=[sig.*par_const sig.*repmat(par_exog,1,size(Q.exo,2))];
elseif demm==1 & demmexo==0
    V_i(:,1)=sig.*par_const; % variance on constant is the hyperparameter times the variance of the AR residual
end
% Now, set the Minnesota variance of the prior distribution on the autoregressive coefficients  
for zz=1:nlags
    for xx = 1:K  % for each equation
        for yy = 1:K   % for each variable and each lag
            if ind(xx,yy)==1 % check if it is a own lag
                Vtemp(xx,yy,zz)=par_ownlag./(zz^par_decay); % variance on own lags, tighter and tighter for higher order lags         
            else Vtemp(xx,yy,zz)=(par_crosslag*sig(xx))./(zz^par_decay*sig(yy)); % variance on cross lags, also tighter and tighter for higher order lags
            end
        end
    end
end

if isempty(lr)==0
    for xx=1:size(lr,1)
        for yy=1:size(lr,2)
            if lr(xx,yy)==1
                lr_bayes(xx,yy)=1;
            elseif lr(xx,yy)==0
                lr_bayes(xx,yy)=0.001;
            end
        end
    end
    for xx=1:nlags
        Vtemp(:,:,xx)=Vtemp(:,:,xx).*lr_bayes;
    end
end

Vtemp=reshape(Vtemp,K,K*nlags);
V_i=[V_i Vtemp];
% Now the variance of the prior distribution of the coefficients is a diagonal matrix with diagonal elements the V_i
varcoeff_prior=V_i(:);
varcoeff_prior=diag(varcoeff_prior);
% Hyperparameters on inv(Q.sigma) ~ W(par1_sigmaprior,inv(par2_sigmaprior))
par1_sigmaprior=0;
par2_sigmaprior=zeros(K,K); 
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Gibbs sampler
%%%%%%%%%%%%%%%%%%

% Initialize the Gibbs sampler
veccoeff_draw=coeff(:);    
coeff_draw=coeff;    
sigma_draw=Q.sigma;

% Now, let's draw the posterior distribution of the parameters of the VAR
% and of the impulse responses
for xx=1:gibbs+burnin  %Start the Gibbs "loop"   
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Draws from the posterior
    
    % Compute posterior variance covariance matrix of coefficients
    % conditional on the draw of the variance covariance matrix of the
    % residuals
    invZZ=inv(Z*Z');
    varcoeff_post=inv(inv(varcoeff_prior) + inv(kron(invZZ,sigma_draw)));
    % Compute posterior mean of coefficients
    % conditional on the draw of the variance covariance matrix of the
    % residuals
    veccoeff_post=varcoeff_post*(inv(varcoeff_prior)*veccoeff_prior+inv(kron(invZZ,sigma_draw))*coeff(:));
    % Draw of the coefficients
    veccoeff_draw=veccoeff_post+chol(varcoeff_post)'*randn(size(veccoeff_post,1),1); 
    coeff_draw=reshape(veccoeff_draw,K,size(coeff,2)); 
    % Compute posterior parameters of the distribution of the variance
    % covariance matrix of the residuals conditional on the draw of the
    % coefficients
    par1_sigmapost=T+par1_sigmaprior;
    par2_sigmapost=par2_sigmaprior+(Y-coeff_draw*Z)*(Y-coeff_draw*Z)';
    % Draw of the variance covariance matrix of the residuals
    sigma_draw=inv(wishrnd(inv(par2_sigmapost),par1_sigmapost));

    % Don't consider the burn period draws 
    if xx>burnin 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Draws from the posterior of the VAR parameters
        coeff_sim(:,:,xx-burnin) = coeff_draw;
        sigma_sim(:,:,xx-burnin) = sigma_draw;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Identification 
        P_temp=chol(sigma_draw)';
        A_temp=zeros(K*nlags,K*nlags);
        if demm==1 & demmexo==1
            A_temp(1:K,:)=coeff_draw(:,2+size(Q.exo,2):end);
        elseif demm==0 & demmexo==0
            A_temp(1:K,:)=coeff_draw;
        elseif demm==1 & demmexo==0
            A_temp(1:K,:)=coeff_draw(:,2:end);
        end
        if nlags>1
            A_temp(K+1:K*nlags,1:K*nlags-K)=eye(K*nlags-K);
        end
        % Find the shock that maximizes forecast error variance of the first variable
        % at horizon H (or cumulative variance from 0 to H if accum=1)
        C_temp=zeros(K*nlags,K);
        C_temp(1:K,1:K)=P_temp;
        e=zeros(K*nlags,1);
        e(1,1)=1;  % Selection vector for first variable
        % Objective function as described in Kurmann and Otrok (2013)
        obj=zeros(K,K,H);
        for yy=1:H
            for zz=1:yy
                if accum==1
                    obj(:,:,yy)=(A_temp^(zz-1)*C_temp)'*e*e'*A_temp^(zz-1)*C_temp+obj(:,:,yy);
                elseif accum==0
                    obj(:,:,yy)=(A_temp^(zz-1)*C_temp)'*e*e'*A_temp^(zz-1)*C_temp;
                end
            end
        end
        
        % Force symmetry to avoid numerical precision issues
        obj(:,:,H) = (obj(:,:,H) + obj(:,:,H)')/2;
        
        [v,D]=eig(obj(:,:,H));
        eigen=diag(D);
        [o,I]=sort(eigen);
        eigen=flipud(o);
        I=flipud(I);
        v=v(:,I');
        % Sign normalization: first shock has positive effect on first variable
        if v(1,1)<0
            v=-v;
        end
        B_temp=P_temp*v;
        B_sim(:,:,xx-burnin)=B_temp;
        
        % Set shock sizes if not provided
        if isempty(shock)==1
            shock_temp=diag(B_temp);
        else
            shock_temp=shock;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Impulse response draw
        Irf_temp=zeros(K*nlags,irf_lenght,K);
        for yy=1:K
            w_temp=zeros(K,1);
            w_temp(yy,1)=shock_temp(yy)/B_temp(yy,yy);
            W_temp=zeros(K*nlags,irf_lenght);
            W_temp(1:K,1)=B_temp*w_temp;
            for zz=1:irf_lenght
                Irf_temp(:,zz+1,yy)=A_temp*Irf_temp(:,zz,yy)+W_temp(:,zz);
            end
        end
        for yy=1:K
            for zz=1:irf_lenght
                % Draws from the posterior impulse response
                irf_sim(:,zz,yy,xx-burnin)=J_mat*Irf_temp(:,zz,yy);
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Variance decomposition draw
        
        C=zeros(K*nlags,K);
        C(1:K,1:K)=B_temp;
        Totvar=zeros(K*nlags,K*nlags,irf_lenght);
        for yy=1:irf_lenght
            for zz=1:yy
                Totvar(:,:,yy)=A_temp^(zz-1)*C*C'*(A_temp^(zz-1))'+Totvar(:,:,yy);
            end
        end
        totvar=zeros(K,K,irf_lenght);
        for yy=1:irf_lenght
            totvar(:,:,yy)=J_mat*Totvar(:,:,yy)*J_mat';
        end
        Var=zeros(K*nlags,K*nlags,irf_lenght,K);
        for yy=1:K
            s=zeros(K,K);
            s(yy,yy)=1;
            for zz=1:irf_lenght
                for hh=1:zz
                    Var(:,:,zz,yy)=A_temp^(hh-1)*C*s*C'*(A_temp^(hh-1))'+Var(:,:,zz,yy);
                end
            end
        end
        var=zeros(K,K,irf_lenght,K);
        for yy=1:K
            for zz=1:irf_lenght
                var(:,:,zz,yy)=J_mat*Var(:,:,zz,yy)*J_mat';
            end
        end
        var_dec=zeros(K,K,irf_lenght,K);
        for yy=1:K
            var_dec(:,:,:,yy)=var(:,:,:,yy)./totvar(:,:,:)*100;
        end
        for yy=1:K
            for zz=1:irf_lenght
                eval(['vardecshock_' shock_names{yy} '_sim(:,zz,xx-burnin)=diag(var_dec(:,:,zz,yy));']);
            end
        end
       
    end 
    
    % Print iteration if verbose
    if verbose==true
        xx
    end
       
end 
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Unconditional posterior mean of the parameters of the VAR

if demm==1 & demmexo==1
    EQ.constant_mean=mean(coeff_sim(:,1,:),3);
    EQ.exo_mean=mean(coeff_sim(:,2:1+size(Q.exo,2),:),3);
    EQ.coefficient_mean=mean(coeff_sim(:,2+size(Q.exo,2):end,:),3); 
elseif demm==0 & demmexo==0
    EQ.coefficient_mean=mean(coeff_sim,3);
elseif demm==1 & demmexo==0
    EQ.constant_mean=mean(coeff_sim(:,1,:),3);
    EQ.coefficient_mean=mean(coeff_sim(:,2:end,:),3); 
end
EQ.coeff_sim=coeff_sim;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Unconditional posterior mean of the identification matrix

EQ.B=mean(B_sim,3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Unconditional posterior impulse responses

perc_irf_up=zeros(K,irf_lenght,K);
irf=zeros(K,irf_lenght,K);
perc_irf_down=zeros(K,irf_lenght,K);
banda1=confianza;
banda2=100-confianza;
for xx=1:K
    for yy=1:irf_lenght
        for hh=1:K
            temp=sort(irf_sim(hh,yy,xx,:));
            temp1=[temp(floor(gibbs*banda2/100));median(temp,4);temp(ceil(gibbs*banda1/100))];
            perc_irf_up(hh,yy,xx)=temp1(1);
            irf(hh,yy,xx)=temp1(2);
            perc_irf_down(hh,yy,xx)=temp1(3);
        end            
    end
end

for xx=1:K
    for yy=1:K
        eval([var_names{yy} '_' shock_names{xx} '=[perc_irf_up(yy,:,xx);irf(yy,:,xx);perc_irf_down(yy,:,xx)];']);
    end
end

for xx=1:K
    count=0;
    figure(xx);
    for yy=1:K
        count=count+1;
        subplot(K,1,count)
        eval(['plot(1:irf_lenght,' var_names{yy} '_' shock_names{xx} '(1,:),'':k'',1:irf_lenght,' var_names{yy} '_' shock_names{xx} '(2,:),''k'',1:irf_lenght,' var_names{yy} '_' shock_names{xx} '(3,:),'':k'');']);
        eval(['title(''' var_names{yy} ' ' shock_names{xx} ''');']);
    end
end

count=0;
for xx=1:K
    for yy=1:K
        count=count+1;
        eval(['EQ.' var_names{yy} '_' shock_names{xx} '=' var_names{yy} '_' shock_names{xx} ';']);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Unconditional posterior mean of the variance decomposition

for yy=1:K
    eval(['vardecshock_' shock_names{yy} '=mean(vardecshock_' shock_names{yy} '_sim,3);']);
end

for xx=1:K
    eval(['EQ.vardecshock_' shock_names{xx} '=vardecshock_' shock_names{xx} ';']);
end

%% Save all IRF draws
EQ.irf_sim=irf_sim;

end