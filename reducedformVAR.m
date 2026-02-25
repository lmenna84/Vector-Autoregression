function EQ=reducedformVAR(data,nlags,const,lr,forecast,dummy,exog)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Author: Lorenzo Menna %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Purpose: 
% Reduced form Vector Autoregression with optional restrictions on
% variables' interactions and forecasting capabilities
% -----------------------------------
% Inputs:
% data = NxK matrix (N number of observations, K number of variables)
% nlags = desired number of lags (no maximum limit enforced)
% Optional:
% const = true if the VAR is estimated with a constant and false if without.
% Default is false.
% lr = KxK matrix of zeros and ones; zeros indicate that the column 
% variable does not affect the row variable, one indicates effect is allowed.
% Default is empty (no restrictions).
% forecast = true to compute out-of-sample forecast, false otherwise. 
% Default is false. Forecast horizon is floor(N/5) periods ahead.
% dummy = 'month' for monthly seasonal dummies (11 dummies) or 'quarter' 
% for quarterly seasonal dummies (3 dummies). Leave empty for no dummies. 
% Default is empty.
% exog = NxE matrix, where E is the number of exogenous variables that
% enter without lags (non-dynamic regressors, e.g., event dummies). 
% These are assumed to be zero in the forecast construction.
% Default is empty.
% -----------------------------------
% Returns:
% EQ = structure with the following fields:
% EQ.constants = Kx1 vector of estimated constants (only if const=true)
% EQ.tstats_const = Kx1 vector of t-statistics for constants (only if const=true)
% EQ.exo = KxE matrix of estimated coefficients for exogenous variables (only if exog provided)
% EQ.tstats_exo = KxE matrix of t-statistics for exogenous variables (only if exog provided)
% EQ.dummies = Kxnumdum matrix of estimated seasonal dummy coefficients (only if dummy provided)
% EQ.tstats_dumm = Kxnumdum matrix of t-statistics for seasonal dummies (only if dummy provided)
% EQ.coefficients = Kx(K*nlags) matrix of estimated VAR coefficients
% EQ.tstats = Kx(K*nlags) matrix of t-statistics for VAR coefficients
% EQ.sigma = KxK variance-covariance matrix of residuals
% EQ.sigma_adj = KxK variance-covariance matrix of residuals adjusted for small sample
% EQ.AIC = Akaike information criterion
% EQ.BIC = Bayesian information criterion
% EQ.resid = (N-nlags)xK matrix of residuals
% EQ.forecast.conf95 = Kx(N+floor(N/5)-1) matrix with 95% confidence bands (only if forecast=true)
% EQ.forecast.cent = Kx(N+floor(N/5)-1) matrix with point forecasts (only if forecast=true)
% EQ.forecast.conf05 = Kx(N+floor(N/5)-1) matrix with 5% confidence bands (only if forecast=true)
% EQ.lastperiod = scalar indicating last period in seasonal cycle (only if dummy provided)
% EQ.data = NxK matrix of original data
% EQ.nlags = number of lags used
% EQ.Sigma = variance-covariance matrix of coefficient estimates
% EQ.Sigma_adj = variance-covariance matrix of coefficient estimates (small-sample adjusted)
% ------------------------------------
% The solution method follows Lütkepohl (2005)

if nargin<7
    exog=[];
end
if nargin<6
    exog=[];
    dummy=[];
end
if nargin<5
    exog=[];
    dummy=[];
    forecast=false;
end
if nargin<4
    exog=[];
    dummy=[];
    forecast=false;
    lr=[];
end
if nargin<3
    exog=[];
    dummy=[];
    forecast=false;
    const=false;
    lr=[];
end
if isempty(const)==1
    const=false;
end
if isempty(forecast)==1
    forecast=false;
end
if isempty(dummy)==1
    numdum=0;
elseif strcmpi('month',dummy)==1
    numdum=11;
elseif strcmpi('quarter',dummy)==1
    numdum=3;
end
if isempty(exog)==1
    numexo=0;
else numexo=size(exog,2);
end

%if nlags>12
 %   display('Mistake:lags bigger than 12');
  %  return
%end

T=size(data,1)-nlags;
K=size(data,2);

if isempty(lr)==0
    if size(lr)~=[K,K]
        display('Mistake: incorrect restrictions');
        return
    end
    for xx=1:size(lr,1)
        for yy=1:size(lr,2)
            if lr(xx,yy)~=0 & lr(xx,yy)~=1
                display('Mistake: incorrect restrictions');
                return
            end
        end
    end
end
    
Y=data(nlags+1:size(data,1),:)';
Zprep=flipud(data(1:size(data,1)-1,:))';
Zprep=reshape(Zprep,(size(data,1)-1)*K,1);
Zprep1=ones(1,T);
Zprepexo=exog(nlags+1:end,:)';

if numdum~=0
    dum=[eye(numdum) zeros(numdum,1)];
    Zprep3=zeros(0,0);
    for xx=1:floor(T/(numdum+1))+1
        Zprep3=[Zprep3 dum];
    end
    Zprep3=Zprep3(:,1:T);
    %for xx=2:size(Zprep3,1)
    for xx=1:size(Zprep3,1)
        if Zprep3(xx,T)==1
            last_period=xx;
        end
    end
    exist last_period;
    if ans==0
        last_period=0;
    end
end
     
Zprep2=zeros(0,0);
count=0;
for xx=1:T
    Zprep2=[Zprep2 Zprep(1+count:nlags*K+count,1)];
    count=count+K;
end

Zprep2=fliplr(Zprep2);

if numdum==0
Z=[Zprep1;Zprepexo;Zprep2];
else Z=[Zprep1;Zprepexo;Zprep3;Zprep2];
end
if numdum==0
clear Zprep Zprep1 Zprep2 Zprepexo
else
    clear Zprep Zprep1 Zprep2 Zprep3 Zprepexo
end

if isempty(lr)==1 & const==true
    M=K*(K*nlags+1)+K*numdum+K*numexo;
    R=eye(M);
elseif isempty(lr)==1 & const==false
    M=K*(K*nlags+1)-K+K*numdum+K*numexo;
    R=zeros(K*(K*nlags+1)+K*numdum+K*numexo,M);
    R(K+1:size(R,1),:)=eye(M);
elseif isempty(lr)==0 & const==true 
    count=0;
    for xx=1:K
        for yy=1:K
        if lr(xx,yy)~=0
            count=count+1;
        end
        end
    end
    M=count*nlags+K+K*numdum+K*numexo;
    R=zeros(K*(K*nlags+1)+K*numdum+K*numexo,M);
    R(1:K+K*numdum+K*numexo,1:K+K*numdum+K*numexo)=eye(K+K*numdum+K*numexo);
    lr1=reshape(lr,K*K,1);
    lr=zeros(0,0);
    for xx=1:nlags
        lr=[lr;lr1];
    end
    count=0;
    for xx=1:size(lr,1)
        if lr(xx,1)==1
        count=count+1;
        R(K+K*numdum+K*numexo+xx,K+K*numdum+K*numexo+count)=1;
        end
    end
elseif isempty(lr)==0 & const==false
    count=0;
    for xx=1:K
        for yy=1:K
        if lr(xx,yy)~=0
            count=count+1;
        end
        end
    end
    M=count*nlags+K*numdum+K*numexo;
    R=zeros(K*(K*nlags+1)+K*numdum+K*numexo,M);
    R(K+1:K+K*numdum+K*numexo,K+1:K+K*numdum+K*numexo)=eye(K*numdum+K*numexo);
    lr1=reshape(lr,K*K,1);
    lr=zeros(0,0);
    for xx=1:nlags
        lr=[lr;lr1];
    end
    count=0;
    for xx=1:size(lr,1)
        if lr(xx,1)==1
        count=count+1;
        R(K+xx,count)=1;
        end
    end
end

%Sigmau_hat=1/(T-K*nlags-1)*Y*(eye(T)-Z'*inv(Z*Z')*Z)*Y';
%Sigmau_hat=1/(T)*Y*(eye(T)-Z'*inv(Z*Z')*Z)*Y';
z=reshape(Y,K*T,1);
%gam_hat=inv(R'*kron(Z*Z',inv(Sigmau_hat))*R)*R'*kron(Z,inv(Sigmau_hat))*z;
gam_onda=inv(R'*kron(Z*Z',eye(K))*R)*R'*kron(Z,eye(K))*z;
bet_onda=R*gam_onda;
B_onda=reshape(bet_onda,K,K*nlags+1+numdum+numexo);
Sigmau_onda=1/(T)*(Y-B_onda*Z)*(Y-B_onda*Z)';
% gam_hat=inv(R'*kron(Z*Z',inv(Sigmau_onda))*R)*R'*kron(Z,inv(Sigmau_onda))*z;
% bet=R*gam_hat;
%  B=reshape(bet,K,K*nlags+1+numdum+numexo);
gam_hat=gam_onda;
bet=bet_onda;
B=B_onda;
Sigmau_hat=Sigmau_onda;

Sigma=R*inv(R'*kron(Z*Z',inv(Sigmau_hat))*R)*R';
Ssigma=reshape(sqrt(diag(Sigma)),K,K*nlags+1+numdum+numexo);
Tstat=B./Ssigma;

AIC=log(det(Sigmau_hat))+2*nlags*K^2/T;
BIC=log(det(Sigmau_hat))+log(T)/T*nlags*K^2;

Uhat=Y-B*Z;
%F=zeros(0,0);
%for xx=1:12
 %   if xx==1
  %      Fxx=eye(T);
   % else Fxx=[zeros(xx-1,T);
    %    eye(T-xx+1) zeros(T-xx+1,xx-1)];
%    end
 %   F=[F Fxx];
  %  UW=kron(eye(xx),Uhat)*F';
   % lambdaLM(xx,1)=reshape(Uhat*UW',K*K*xx,1)'*kron(inv(UW*UW'-UW*Z'*inv(Z*Z')*Z*UW'),inv(Sigmau_hat))*reshape(Uhat*UW',K*K*xx,1);
% end

if const==true
    EQ.constants=B(:,1);
    EQ.tstats_const=Tstat(:,1);
end
if isempty(exog)==0
    EQ.exo=B(:,2:numexo+1);
    EQ.tstats_exo=Tstat(:,2:numexo+1);
end
if isempty(dummy)==0
    EQ.dummies=B(:,2+numexo:numdum+numexo+1);
    EQ.tstats_dumm=Tstat(:,2+numexo:numdum+numexo+1);
end
EQ.coefficients=B(:,2+numdum+numexo:size(B,2));
EQ.tstats=Tstat(:,2+numdum+numexo:size(Tstat,2));
EQ.sigma=Sigmau_hat;
EQ.sigma_adj=T/(T-K*nlags-1)*Sigmau_hat;
EQ.AIC=AIC;
EQ.BIC=BIC;
EQ.resid=Uhat';
% EQ.LM=lambdaLM;
A=zeros(K*nlags,K*nlags);
A(1:K,:)=B(:,2+numdum+numexo:size(B,2));
J=zeros(K,K*nlags);
J(:,1:K)=eye(K);
if nlags>1
    A(K+1:K*nlags,1:K*nlags-K)=eye(K*nlags-K);
end
%EQ.roots=eig(A);
clear A J

if numdum~=0
    EQ.lastperiod=last_period;
end
EQ.data=data;
EQ.nlags=nlags;
EQ.Sigma=Sigma;
EQ.Sigma_adj=R*inv(R'*kron(Z*Z',inv(EQ.sigma_adj))*R)*R';

% Forecasting
if forecast==true
    leng=floor(size(data,1)/5);
    V=zeros(K*nlags,1+numdum);
    if const==true
        V(1:K,1)=B(:,1);
    end
    if isempty(dummy)==0
        V(1:K,2:numdum+const)=B(:,2+numexo:const+numexo+numdum);
    end
    A=zeros(K*nlags,K*nlags);
    A(1:K,:)=B(:,2+numexo+numdum:size(B,2));
    J=zeros(K,K*nlags);
    J(:,1:K)=eye(K);
    if nlags>1
        A(K+1:K*nlags,1:K*nlags-K)=eye(K*nlags-K);
    end
    Pron=zeros(K*nlags,leng);
    %Pron(1:K,1)=Y(:,T);
    Pron(1:K*nlags,1)=reshape(flip(Y(:,T-nlags+1:T),2),K*nlags,1);
    conta=0;
    for yy=1:leng-1
        conta=conta+1;
        if isempty(dummy)==0
        vec_dummies=zeros(size(V,2)-1,1);
        if mod(last_period+conta,numdum+1)~=0
            prot=mod(last_period+conta,numdum+1);
            vec_dummies(prot,1)=1;
        end
        vec_dummies=[1;vec_dummies];
        else vec_dummies=1;
        end
        Pron(:,yy+1)=V*vec_dummies+A*Pron(:,yy); 
    end
    for yy=1:leng
        pron(:,yy)=J*Pron(:,yy);
    end
    u=randn(K,leng,1000);
    for xx=1:leng
        for yy=1:1000
            u(:,xx,yy)=chol(Sigmau_hat)'*u(:,xx,yy);
        end
    end
    U=zeros(K*nlags,leng,1000);
    U(1:K,:,:)=u;
    Pron_conf=zeros(K*nlags,leng,1000);
    for xx=1:1000
        Pron_conf(1:K*nlags,1,xx)=reshape(flip(Y(:,T-nlags+1:T),2),K*nlags,1);
    end
    for xx=1:1000
        conta=0;
        for yy=1:leng-1
            conta=conta+1;
             if isempty(dummy)==0
                vec_dummies=zeros(size(V,2)-1,1);
                if mod(last_period+conta,numdum+1)~=0
                prot=mod(last_period+conta,numdum+1);
                vec_dummies(prot,1)=1;
                end
                vec_dummies=[1;vec_dummies];
                else vec_dummies=1;
                end
            Pron_conf(:,yy+1,xx)=V*vec_dummies+A*Pron_conf(:,yy,xx)+U(:,yy+1,xx);
        end
    end
    for xx=1:1000
        for yy=1:leng
            pron_conf(:,yy,xx)=J*Pron_conf(:,yy,xx);
        end
    end
    pron_up=zeros(K,leng);
    pron_down=zeros(K,leng);
    for xx=1:K
        for yy=1:leng
            temp=sort(pron_conf(xx,yy,:));
            temp1=[temp(1000*95/100);temp(1000*5/100)];
            pron_up(xx,yy)=temp1(1);
            pron_down(xx,yy)=temp1(2);
        end
    end
    
    FULL=[Y(:,1:end-1) pron];
    FULL_UP=[Y(:,1:end-1) pron_up];
    FULL_DOWN=[Y(:,1:end-1) pron_down];
    
    for xx=1:size(data,2)
        eval(['var_names{' int2str(xx) '}=''var' int2str(xx) ''';']);
    end
    
    for xx=1:K
    figure(xx);
    plot(1:T+leng-1,FULL_UP(xx,:),':k',...
        1:T+leng-1,FULL(xx,:),'k',...
        1:T+leng-1,FULL_DOWN(xx,:),':k');
    title(var_names{xx});
    end
    EQ.forecast.conf95=FULL_UP;
    EQ.forecast.cent=FULL;
    EQ.forecast.conf05=FULL_DOWN;
    
end



end