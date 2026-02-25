function EQ=signrestSVAR(data,nlags,sign,periods_sign,var_names,shock_names,irf_lenght,repetitions,const,lr,dum,exog,shock,confianza,vardec,verbose)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Author: Lorenzo Menna %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Purpose: 
% Impulse responses and Variance decomposition of a Structural Vector Autoregression
% with sign restrictions (Uhlig 2005, Rubio-Ramirez et al. 2010).
% 
% This function uses uninformative priors and focuses on identification uncertainty
% rather than parameter uncertainty.
% -----------------------------------
% Inputs:
% data = NxK matrix (N number of observations, K number of variables)
% nlags = desired number of lags (no maximum limit enforced)
% sign = KxK matrix with 1 when a positive contemporaneous response is
%        desired, -1 for a negative response and 0 for no restriction.
%        Element (i,j) is the sign of the response of variable i to shock j.
% Optional:
% periods_sign = scalar. Number of periods in which the restrictions have
%                to be enforced. Default is 1 (only impact restrictions).
% var_names = 1xK cell vector with names of the variables
% shock_names = 1xK cell vector with names of the shocks
% irf_lenght = desired length of the IRFs. Default = 40.
% repetitions = number of repetitions for the posterior distribution.
%               Default=1000
% const = true if the VAR is estimated with a constant and false if without.
%         Default is false.
% lr = KxK matrix of zeros and ones specifying restrictions on the reduced-form VAR.
%      Element (i,j) = 0 restricts variable j from affecting variable i.
%      Default is no restrictions.
% dum = seasonal dummies. Specify 'month' for 11 monthly dummies or 
%       'quarter' for 3 quarterly dummies. Default is no dummies.
% exog = NxE matrix, where E is the number of exogenous variables that
%        enter without lags (non-dynamic regressors, e.g., event dummies).
% shock = Kx1 vector containing the size of the shock for each variable. 
%         Default is one standard deviation shocks.
% confianza = level in % of the lowest confidence band. Default is 5%.
% vardec = scalar equal to 1 if one wants to compute the variance
%          decomposition, 0 otherwise. Default is zero.
% verbose = true to print iteration numbers, false to suppress. Default is false.
% -----------------------------------
% Returns:
% EQ = structure with the following fields:
% EQ.vari_shockj = K*K 3xirf_lenght matrices containing in the first row the
%                  95% confidence line, in the second row the median IRF, and in 
%                  the third row the 5% confidence line. i is the variable, j is the shock.
% EQ.vardecshock_shockj = K Kxirf_lenght matrices containing in each row j the
%                         percentage of variance of variable j due to shock i. 
%                         Each column is the number of periods ahead. Median variance decomposition.
% EQ.irfsim = Kxirf_lenghtxrepetitionsxK array of all IRF draws
% ------------------------------------
% The solution method follows Lütkepohl (2005), Uhlig (2005), and 
% Rubio-Ramirez, Waggoner, and Zha (2010)

if nargin<16
    verbose=false;
end
if nargin<15
    verbose=false;
    vardec=0;
end
if nargin<14
    verbose=false;
    vardec=0;
    confianza=5;
end
if nargin<13
    verbose=false;
    vardec=0;
    confianza=5;
    shock=[];
end
if nargin<12
    verbose=false;
    vardec=0;
    confianza=5;
    shock=[];
    exog=[];
end
if nargin<11
    verbose=false;
    vardec=0;
    confianza=5;
    shock=[];
    exog=[];
    dum=[];
end
if nargin<10
    verbose=false;
    vardec=0;
    confianza=5; 
    shock=[];
    exog=[];
    dum=[];
    lr=[];
end
if nargin<9
    verbose=false;
    vardec=0;
    confianza=5; 
    shock=[];
    exog=[];
    dum=[];
    const=false;
    lr=[];
end
if nargin<8
    verbose=false;
    vardec=0;
    confianza=5; 
    shock=[];
    exog=[];
    dum=[];
    const=false;
    lr=[];
    repetitions=1000;
end
if nargin<7
    verbose=false;
    vardec=0;
    confianza=5; 
    shock=[];
    exog=[];
    dum=[];
    const=false;
    lr=[];
    repetitions=1000;
    irf_lenght=40;
end
if nargin<6
    verbose=false;
    vardec=0;
    confianza=5; 
    shock=[];
    exog=[];
    dum=[];
    const=false;
    lr=[];
    repetitions=1000;
    irf_lenght=40;
    for xx=1:size(data,2)
        eval(['shock_names{' int2str(xx) '}=''var' int2str(xx) ''';']);
    end
end
if nargin<5
    verbose=false;
    vardec=0;
    confianza=5; 
    shock=[];
    exog=[];
    dum=[];
    const=false;
    lr=[];
    repetitions=1000;
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
    vardec=0;
    confianza=5; 
    shock=[];
    exog=[];
    dum=[];
    const=false;
    lr=[];
    repetitions=1000;
    irf_lenght=40;
    for xx=1:size(data,2)
        eval(['shock_names{' int2str(xx) '}=''var' int2str(xx) ''';']);
    end
    for xx=1:size(data,2)
        eval(['var_names{' int2str(xx) '}=''var' int2str(xx) ''';']);
    end
    periods_sign=1;
end

if isempty(verbose)==1
    verbose=false;
end
if isempty(vardec)==1
    vardec=0;
end
if isempty(shock)==1
    shock=[];
end
if isempty(repetitions)==1
    repetitions=1000;
end
if isempty(irf_lenght)==1
    irf_lenght=40;
end
if isempty(confianza)==1
    confianza=5;
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
if isempty(periods_sign)==1
    periods_sign=1;
end

K=size(data,2);
T=size(data,1)-nlags;
RF=reducedformVAR(data,nlags,const,lr,[],dum,exog);
demm=isfield(RF,'constants');
demmexo=isfield(RF,'exo');
if demm==1 & demmexo==1
    coeff=[RF.constants RF.exo RF.coefficients];
elseif demm==0 & demmexo==0
    coeff=[RF.coefficients];
elseif demm==1 & demmexo==0
    coeff=[RF.constants RF.coefficients];
end
sigma=RF.sigma;
J_mat=zeros(K,K*nlags);
J_mat(:,1:K)=eye(K);

zz=0;
for ii=1:repetitions
    if verbose==true
        ii
    end
    % Assume that the varcov matrix is inverse Wishart with df=T and obtain draw
    inv_sigma=inv(sigma);
    inv_sigma_draw=wishrnd(inv_sigma/T,T);
    sigma_draw=inv(inv_sigma_draw);
    
    % Generate regressors: they are needed to obtain variance of the coefficients
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
    
    % Coefficient variance is the kronecker of varcov matrix with inverse of Z*Z'
    invZZ=inv(Z*Z');
    var_coeff=kron(invZZ,sigma_draw);
    
    % vectorize coefficients
    vec_coeff=coeff(:);
    
    % Obtain draw
    coeff_draw=mvnrnd(vec_coeff,var_coeff);
    if demmexo==0
        coeff_draw=reshape(coeff_draw,K,const+K*nlags);
    else coeff_draw=reshape(coeff_draw,K,const+size(exog,2)+K*nlags);
    end
    
    % Now obtain Cholesky
    P=chol(sigma_draw)';
    
    ww=0;
    jj=0;
    while jj<1
        % Generate candidate structural shock
        ww=ww+1;
        if verbose==true
            ww
        end
        W=randn(K,K);
        [Q,R]=qr(W);
        % Diagonal of R has to be positive (see Lütkepohl), so flip sign of row if it's not
        for xx=1:K
            if R(xx,xx)<0
                R(xx,:)=-R(xx,:);
            end
        end
        % Then ensure that Q*R=W is still satisfied
        Q=W*inv(R);
        % Find the impact multiplier
        B=P*Q;
        
        % Column permutation algorithm: find best permutation of B columns
        % to match sign pattern while preserving orthogonality
        Bnew=zeros(size(B));
        for yy=1:K
            xx=0;
            prova=[1 -1];
            while all(prova>=0)==0 & all(prova<=0)==0 & xx<=size(B,2)-1
                xx=xx+1;
                prova=B(:,xx).*sign(:,yy);  
            end
            if xx<=size(B,2)-1
                if all(prova>=0)==1
                    Bnew(:,yy)=B(:,xx);
                else Bnew(:,yy)=-B(:,xx);
                end
                B(:,xx)=[];
            else Bnew(:,yy)=B(:,1);
                B(:,1)=[];
            end
        end
        B=Bnew;
        
        % The sign of prova is positive if requested sign corresponds to actual sign
        prova=B.*sign;
        if all(all(prova>=0)==1)==1 & periods_sign>1
            % If multi-period restrictions, check signs at all horizons
            A_temp=zeros(K*nlags,K*nlags);
            if demmexo==0
            A_temp(1:K,:)=coeff_draw(:,1+demm:end);
            else A_temp(1:K,:)=coeff_draw(:,1+demm+size(exog,2):end);
            end
            if nlags>1
                A_temp(K+1:K*nlags,1:K*nlags-K)=eye(K*nlags-K);
            end
            Irf_temp=zeros(K*nlags,periods_sign,K);
            % Compute impulse response
            for xx=1:K
                w_temp=zeros(K,1);
                w_temp(xx,1)=1;
                W_temp=zeros(K*nlags,periods_sign);
                W_temp(1:K,1)=B*w_temp;
                for yy=1:periods_sign
                    Irf_temp(:,yy+1,xx)=A_temp*Irf_temp(:,yy,xx)+W_temp(:,yy);
                    irf_sim(:,yy+1,xx)=J_mat*Irf_temp(:,yy+1,xx);
                end
            end
            for xx=1:periods_sign
                for yy=1:K
                    B_periods(:,yy,xx)=irf_sim(:,xx+1,yy);
                end
            end
            prova_periods=B_periods.*sign;
        else prova_periods=prova;
        end  
        if all(all(all(prova_periods>=0)==1))==1
            jj=jj+1;
            A_temp=zeros(K*nlags,K*nlags);
            if demmexo==0
            A_temp(1:K,:)=coeff_draw(:,1+demm:end);
            else A_temp(1:K,:)=coeff_draw(:,1+demm+size(exog,2):end);
            end
            if nlags>1
                A_temp(K+1:K*nlags,1:K*nlags-K)=eye(K*nlags-K);
            end
            Irf_temp=zeros(K*nlags,irf_lenght,K);
            
            % Set shock sizes if not provided
            if isempty(shock)==1
                shock_temp=diag(B);
            else
                shock_temp=shock;
            end
            
            % Compute impulse response
            for xx=1:K
                w_temp=zeros(K,1);
                w_temp(xx,1)=shock_temp(xx)/B(xx,xx);
                W_temp=zeros(K*nlags,irf_lenght);
                W_temp(1:K,1)=B*w_temp;
                for yy=1:irf_lenght
                    Irf_temp(:,yy+1,xx)=A_temp*Irf_temp(:,yy,xx)+W_temp(:,yy);
                    irf_sim(:,yy,zz+1,xx)=J_mat*Irf_temp(:,yy,xx);
                end
            end
            
            if vardec==1
            % Compute variance decomposition
            C=zeros(K*nlags,K);
            C(1:K,1:K)=B;
            Totvar=zeros(K*nlags,K*nlags,irf_lenght);
            for xx=1:irf_lenght
                for yy=1:xx
                    Totvar(:,:,xx)=A_temp^(yy-1)*C*C'*(A_temp^(yy-1))'+Totvar(:,:,xx);
                end
            end
            totvar=zeros(K,K,irf_lenght);
            for xx=1:irf_lenght
                totvar(:,:,xx)=J_mat*Totvar(:,:,xx)*J_mat';
            end
            Var=zeros(K*nlags,K*nlags,irf_lenght,K);
            for xx=1:K
                s=zeros(K,K);
                s(xx,xx)=1;
                for yy=1:irf_lenght
                    for hh=1:yy
                        Var(:,:,yy,xx)=A_temp^(hh-1)*C*s*C'*(A_temp^(hh-1))'+Var(:,:,yy,xx);
                    end
                end
            end
            var=zeros(K,K,irf_lenght,K);
            for xx=1:K
                for yy=1:irf_lenght
                var(:,:,yy,xx)=J_mat*Var(:,:,yy,xx)*J_mat';
                end
            end
            var_dec=zeros(K,K,irf_lenght,K);
            for xx=1:K
                var_dec(:,:,:,xx)=var(:,:,:,xx)./totvar(:,:,:)*100;
            end   
            for xx=1:K
                for yy=1:irf_lenght
                eval(['vardecshock_' shock_names{xx} '(:,yy,zz+1)=diag(var_dec(:,:,yy,xx));']);
                end
            end
            end
        end
        if ww>1000
            break
            display('Too few models satisfy the sign restrictions')
        end
    end
    zz=zz+jj;
end

perc_irf_up=zeros(K,irf_lenght,K);
perc_irf_down=zeros(K,irf_lenght,K);
banda1=confianza;
banda2=100-confianza;
for xx=1:K
    for yy=1:irf_lenght
        for hh=1:K
            temp=sort(irf_sim(hh,yy,:,xx));
            temp1=[temp(floor(repetitions*banda2/100));median(temp);temp(ceil(repetitions*banda1/100))];
            perc_irf_up(hh,yy,xx)=temp1(1);
            irf(hh,yy,xx)=temp1(2);
            perc_irf_down(hh,yy,xx)=temp1(3);
        end            
    end
end

if vardec==1
for xx=1:K
    for yy=1:irf_lenght
        for hh=1:K
            eval(['temp=sort(vardecshock_' shock_names{xx} '(hh,yy,:));']);
            eval(['central_vardecshock_' shock_names{xx} '(hh,yy)=median(temp);']);
        end            
    end
end
end

EQ.irfsim=irf_sim;

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

if vardec==1
for xx=1:K
    eval(['EQ.vardecshock_' shock_names{xx} '=central_vardecshock_' shock_names{xx} ';']);
end
end

end