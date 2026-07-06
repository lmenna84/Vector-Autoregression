function EQ=bayes_ivSVAR(data_inst,other_data,iv,nlags,var_names,shock_names,irf_lenght,gibbs,const,lr,exog,confianza,parbayes,burnin,verbose)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Author: Lorenzo Menna %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Purpose:
% Bayesian IV-SVAR (proxy-SVAR) with plug-in identification following
% Miranda-Agrippino & Ricco (2021, AEJ:Macro). Minnesota Independent
% Normal-Wishart prior on the reduced-form VAR, Mertens & Ravn (2013)
% IV identification applied at each Gibbs draw.
% -----------------------------------
% Inputs:
% data_inst = NxK1 matrix of instrumented variables (ordered first in the VAR)
% other_data = NxK2 matrix of remaining VAR variables
% iv = Nx1 vector of external instrument (may contain NaN for missing obs)
% nlags = number of lags
% var_names = 1xK cell with variable names (K = K1+K2)
% shock_names = 1xK1 cell with shock names
% irf_lenght = IRF horizon (default 40)
% gibbs = number of Gibbs sampler draws (default 2000)
% const = include constant (default false)
% lr = KxK matrix of 0/1 restrictions on lags; also used for contemporaneous
%       IV restrictions (zeros in first K1 columns block non-instrumented effects)
% exog = NxE exogenous regressors entering without lags
% confianza = confidence band percentile (default 16)
% parbayes = 1x6 Minnesota prior hyperparameters [alf1 alf2 alf3 decay alf4 alf5]
% burnin = burn-in draws (default 500)
% verbose = print iterations (default false)
% -----------------------------------
% Returns:
% EQ.vari_shockj = 3xirf_lenght matrices (upper band, median, lower band)
% EQ.P = KxK1 posterior mean of the impact matrix
% EQ.robustF = first-stage robust F-statistic (from OLS)
% EQ.vardecshock_shockj = Kxirf_lenght variance decomposition for identified shocks
% EQ.coefficient_mean, EQ.constant_mean, EQ.exo_mean
% EQ.coeff_sim, EQ.irf_sim

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% beginnings...
%%%%%%%%%%%%%%%%%%%%%%
if nargin<15
    verbose=false;
end
if nargin<14
    verbose=false;
    burnin=500;
end
if nargin<13
    verbose=false;
    burnin=500;
    parbayes=[];
end
if nargin<12
    verbose=false;
    burnin=500;
    parbayes=[];
    confianza=16;
end
if nargin<11
    verbose=false;
    burnin=500;
    parbayes=[];
    confianza=16;
    exog=[];
end
if nargin<10
    verbose=false;
    burnin=500;
    parbayes=[];
    confianza=16;
    exog=[];
    lr=[];
end
if nargin<9
    verbose=false;
    burnin=500;
    parbayes=[];
    confianza=16;
    exog=[];
    const=false;
    lr=[];
end
if nargin<8
    verbose=false;
    burnin=500;
    parbayes=[];
    confianza=16;
    exog=[];
    const=false;
    lr=[];
    gibbs=2000;
end
if nargin<7
    verbose=false;
    burnin=500;
    parbayes=[];
    confianza=16;
    exog=[];
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
    exog=[];
    const=false;
    lr=[];
    gibbs=2000;
    irf_lenght=40;
    shock_names={};
end
if nargin<5
    verbose=false;
    burnin=500;
    parbayes=[];
    confianza=16;
    exog=[];
    const=false;
    lr=[];
    gibbs=2000;
    irf_lenght=40;
    shock_names={};
    var_names={};
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
if isempty(gibbs)==1
    gibbs=2000;
end
if isempty(irf_lenght)==1
    irf_lenght=40;
end
if isempty(confianza)==1
    confianza=16;
end

K1=size(data_inst,2);
K2=size(other_data,2);
data=[data_inst, other_data];
K=size(data,2);

if isempty(var_names)==1
    for xx=1:K
        eval(['var_names{' int2str(xx) '}=''var' int2str(xx) ''';']);
    end
end
if isempty(shock_names)==1
    for xx=1:K1
        eval(['shock_names{' int2str(xx) '}=''shock' int2str(xx) ''';']);
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Start with OLS estimation of the VAR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

T=size(data,1)-nlags;
Q=reducedformVAR(data,nlags,const,lr,[],[],exog);

% Compute robust F-statistic from OLS residuals
resid_ols=Q.resid;
resid1_ols=resid_ols(:,1:K1);
iv_trimmed=iv(nlags+1:end,:);
valid_iv=all(~isnan(iv_trimmed),2);
iv_valid=iv_trimmed(valid_iv,:);
n_valid=sum(valid_iv);

for xx=1:K1
    X_fs=iv_valid;
    y_fs=resid1_ols(valid_iv,xx);
    beta_fs=X_fs\y_fs;
    resid_fs=y_fs-X_fs*beta_fs;
    meat=X_fs'*diag(resid_fs.^2)*X_fs;
    bread=inv(X_fs'*X_fs);
    vcov_robust=bread*meat*bread;
    SE_robust=sqrt(diag(vcov_robust))';
    EQ.robustF(xx,:)=(beta_fs'./SE_robust).^2;
end
fprintf('Robust F-statistic: %.4f\n', EQ.robustF);

% Put all coefficients in a single matrix
demm=isfield(Q,'constants');
demmexo=isfield(Q,'exo');
if demm==1 & demmexo==1
    coeff=[Q.constants Q.exo Q.coefficients];
elseif demm==0 & demmexo==0
    coeff=[Q.coefficients];
elseif demm==1 & demmexo==0
    coeff=[Q.constants Q.coefficients];
end

% Build matrix of independent variables and dependent for use in the Bayesian part
Zprep=flipud(data(1:size(data,1)-1,:))';
Zprep=reshape(Zprep,(size(data,1)-1)*K,1);
if demm==1
    Zprep1=ones(1,T);
else Zprep1=[];
end
if isempty(exog)
    Zprepexo=[];
else
    Zprepexo=exog(nlags+1:end,:)';
end
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

coeff_prior=zeros(size(coeff));
if demm==1 & demmexo==1
    coeff_prior(:,1+size(Q.exo,2)+1:1+size(Q.exo,2)+K)=alf1*eye(K);
elseif demm==0 & demmexo==0
    coeff_prior(:,1:K)=alf1*eye(K);
elseif demm==1 & demmexo==0
    coeff_prior(:,2:1+K)=alf1*eye(K);
end
veccoeff_prior=coeff_prior(:);

par_ownlag = alf2;
par_crosslag = alf3*par_ownlag;
par_decay = decay;
if demm==1
    par_const = alf4;
end
if demmexo==1
    par_exog = alf5;
end

for xx=1:K
    RF=reducedformVAR(data(:,xx),nlags,0);
    sig(xx,1)=RF.sigma;
end

ind=eye(K);
for zz=1:nlags
    for xx = 1:K
        for yy = 1:K
            if ind(xx,yy)==1
                Vtemp(xx,yy,zz)=par_ownlag./(zz^par_decay);
            else Vtemp(xx,yy,zz)=(par_crosslag*sig(xx))./(zz^par_decay*sig(yy));
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
V_i=[];
if demm==1 & demmexo==1
    V_i=[sig.*par_const sig.*repmat(par_exog,1,size(Q.exo,2))];
elseif demm==1 & demmexo==0
    V_i=sig.*par_const;
end
V_i=[V_i Vtemp];

varcoeff_prior=V_i(:);
varcoeff_prior=diag(varcoeff_prior);
par1_sigmaprior=0;
par2_sigmaprior=zeros(K,K);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Gibbs sampler
%%%%%%%%%%%%%%%%%%

% Initialize the Gibbs sampler
veccoeff_draw=coeff(:);
coeff_draw=coeff;
sigma_draw=Q.sigma;
P_prev=zeros(K,K1);
J=zeros(K,K*nlags);
J(:,1:K)=eye(K);

% Now, let's draw the posterior distribution of the parameters of the VAR
% and of the impulse responses
for xx=1:gibbs+burnin  %Start the Gibbs "loop"
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Draws from the posterior

    % Compute posterior variance covariance matrix of coefficients
    % conditional on the draw of the variance covariance matrix of the
    % residuals (numerically stable formulation)
    inv_prior=inv(varcoeff_prior);
    kron_term=kron(Z*Z', inv(sigma_draw));
    M_post=inv_prior + kron_term;
    M_post=(M_post + M_post')/2;  % force symmetry
    varcoeff_post=inv(M_post);
    varcoeff_post=(varcoeff_post + varcoeff_post')/2;  % force symmetry
    % Compute posterior mean of coefficients
    % conditional on the draw of the variance covariance matrix of the
    % residuals
    veccoeff_post=varcoeff_post*(inv_prior*veccoeff_prior+kron_term*coeff(:));
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
        %% IV Identification (plug-in Mertens & Ravn)
        u_draw=(Y-coeff_draw*Z)';  % T x K residuals from this draw
        resid1_draw=u_draw(:,1:K1);
        resid2_draw=u_draw(:,K1+1:K);

        % First stage: regress instrumented residuals on instruments
        pred_fs=zeros(n_valid,K1);
        for jj=1:K1
            X_fs=iv_trimmed(valid_iv,:);
            y_fs=resid1_draw(valid_iv,jj);
            beta_fs=X_fs\y_fs;
            pred_fs(:,jj)=X_fs*beta_fs;
        end

        % Second stage: effects on non-instrumented variables
        X_mr=zeros(K2,K1);
        for jj=1:K2
            y_ss=resid2_draw(valid_iv,jj);
            beta_ss=pred_fs\y_ss;
            X_mr(jj,:)=beta_ss';
        end

        % Apply contemporaneous restrictions from lr
        if ~isempty(lr)
            for jj=1:K2
                for kk=1:K1
                    if lr(K1+jj,kk)==0
                        X_mr(jj,kk)=0;
                    end
                end
            end
        end

        X_mr=X_mr(:,1);

        % Mertens-Ravn algebra using sigma_draw
        varcovar11=sigma_draw(1:K1,1:K1);
        varcovar21=sigma_draw(K1+1:K,1:K1);
        varcovar22=sigma_draw(K1+1:K,K1+1:K);

        Z_mr=X_mr*varcovar11*X_mr'-(varcovar21*X_mr'+X_mr*varcovar21')+varcovar22;
        bet12_bet12t=(varcovar21-X_mr*varcovar11)'*inv(Z_mr)*(varcovar21-X_mr*varcovar11);
        bet11_bet11t=varcovar11-bet12_bet12t;
        bet22_bet22t=varcovar22+X_mr*(bet12_bet12t-varcovar11)*X_mr';
        bet12_inv_bet22=(bet12_bet12t*X_mr'+(varcovar21-X_mr*varcovar11)')*inv(bet22_bet22t);

        S1_S1t=(eye(K1)-bet12_inv_bet22*X_mr)*bet11_bet11t*(eye(K1)-bet12_inv_bet22*X_mr)';

        try
            S1=chol(S1_S1t)';
            bet11=inv(eye(K1)-bet12_inv_bet22*X_mr)*S1;
            bet21=X_mr*inv(eye(K1)-bet12_inv_bet22*X_mr)*S1;
            P_temp=[bet11; bet21];
            P_prev=P_temp;
        catch
            P_temp=P_prev;  % use previous successful identification
        end

        P_sim(:,:,xx-burnin)=P_temp;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Impulse response draw for identified shocks
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
        Irf_temp=zeros(K*nlags,irf_lenght,K1);
        for yy=1:K1
            W_temp=zeros(K*nlags,irf_lenght);
            W_temp(1:K,1)=P_temp(:,yy);
            for zz=1:irf_lenght
                Irf_temp(:,zz+1,yy)=A_temp*Irf_temp(:,zz,yy)+W_temp(:,zz);
            end
        end
        for yy=1:K1
            for zz=1:irf_lenght
                irf_sim(:,zz,yy,xx-burnin)=J*Irf_temp(:,zz,yy);
            end
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Variance decomposition for identified shocks

        % Total forecast error variance from the full sigma
        C_full=zeros(K*nlags,K);
        C_full(1:K,1:K)=chol(sigma_draw)';
        Totvar=zeros(K*nlags,K*nlags,irf_lenght);
        count=0;
        for yy=1:irf_lenght
            count=count+1;
            for zz=1:count
                Totvar(:,:,yy)=A_temp^(zz-1)*C_full*C_full'*(A_temp^(zz-1))'+Totvar(:,:,yy);
            end
        end
        totvar=zeros(K,K,irf_lenght);
        for yy=1:irf_lenght
            totvar(:,:,yy)=J*Totvar(:,:,yy)*J';
        end

        % Per-shock contribution for each identified shock
        for yy=1:K1
            col_yy=zeros(K*nlags,1);
            col_yy(1:K)=P_temp(:,yy);
            Var_yy=zeros(K*nlags,K*nlags,irf_lenght);
            count=0;
            for zz=1:irf_lenght
                count=count+1;
                for hh=1:count
                    Var_yy(:,:,zz)=A_temp^(hh-1)*col_yy*col_yy'*(A_temp^(hh-1))'+Var_yy(:,:,zz);
                end
            end
            for zz=1:irf_lenght
                var_yy_temp=J*Var_yy(:,:,zz)*J';
                eval(['vardecshock_' shock_names{yy} '_sim(:,zz,xx-burnin)=diag(var_yy_temp)./diag(totvar(:,:,zz))*100;']);
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

EQ.P=mean(P_sim,3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Unconditional posterior impulse responses

perc_irf_up=zeros(K,irf_lenght,K1);
irf=zeros(K,irf_lenght,K1);
perc_irf_down=zeros(K,irf_lenght,K1);
banda1=confianza;
banda2=100-confianza;
for xx=1:K1
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

for xx=1:K1
    for yy=1:K
        eval([var_names{yy} '_' shock_names{xx} '=[perc_irf_up(yy,:,xx);irf(yy,:,xx);perc_irf_down(yy,:,xx)];']);
    end
end

for xx=1:K1
    figure(xx);
    count=0;
    for yy=1:K
        count=count+1;
        subplot(K,1,count)
        eval(['plot(1:irf_lenght,' var_names{yy} '_' shock_names{xx} '(1,:),'':k'',1:irf_lenght,' var_names{yy} '_' shock_names{xx} '(2,:),''k'',1:irf_lenght,' var_names{yy} '_' shock_names{xx} '(3,:),'':k'');']);
        eval(['title(''' var_names{yy} ' ' shock_names{xx} ''');']);
    end
end

count=0;
for xx=1:K1
    for yy=1:K
        count=count+1;
        eval(['EQ.' var_names{yy} '_' shock_names{xx} '=' var_names{yy} '_' shock_names{xx} ';']);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Unconditional posterior mean of the variance decomposition

for yy=1:K1
    eval(['vardecshock_' shock_names{yy} '=mean(vardecshock_' shock_names{yy} '_sim,3);']);
end

for xx=1:K1
    eval(['EQ.vardecshock_' shock_names{xx} '=vardecshock_' shock_names{xx} ';']);
end

%% Save all IRF draws
EQ.irf_sim=irf_sim;

end
