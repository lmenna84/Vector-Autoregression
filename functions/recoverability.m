function Q=recoverability(data,iv,nlags,nleads,nlagsclean,const,parlags,controls)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Author: Lorenzo Menna %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Purpose: 
% Forni et al recoverability test. The Null hipothesis is recoverability
% -----------------------------------
% Inputs:
% data = NxK matrix (N number of observations, K number of variables in VAR). 
% iv = Nx1 vector of instrumental variable.
% nlags = desired number of lags 
% nleads = desired number of leads for test
% nlagsclean = number of lags in the clening equation. Cannot be larger
% than nlags
% Optional:
% const = true if the VAR is estimated with a constant and = false if without.
% Default is without.
% parlags = number of lags in the Ljung Box test. Default is sqrt(size(data,1)-nleads)
% controls = NxZ matrix, Z number of control variables to include in
% instrumental variable regression
% -----------------------------------
% Returns:
% Q = structure
% Q.LBQ = Ljung Box statistics
% Q.p = p value of the test
% Q.BICclean = BIC of the regressin that cleans the instrument
% Q.BICwold = BIC of the regression of the cleaned instrument vs residuals
% ------------------------------------

if nargin<8
    controls=[];
end
if nargin<7
    controls=[];
    parlags=floor(sqrt(size(data,1)-nleads));
end
if nargin<6
    controls=[];
    parlags=floor(sqrt(size(data,1)-nleads));
    const=0;
end

if isempty(parlags)==1
    parlags=floor(sqrt(size(data,1)-nleads));
end

K=size(data,2);
Z=size(controls,2);
% Clean instrument
if nlagsclean==0
    qq.u=iv(nlags+1:end);
else lr=[zeros(K+Z+1,1) ones(K+Z+1,nlagsclean)];
    qq=distlagOLS([iv data controls],nlagsclean,const,lr);
    Q.BICclean=qq.BIC;
    qq.u=qq.u(1+(nlags-nlagsclean):end);
end
% Reduced form VAR
RF=reducedformVAR(data,nlags,const);
% Regress on Wald residuals
cleaninst=qq.u(1:end-nleads);
waldres=[RF.resid(1:end-nleads,:)];
for xx=1:nleads
    waldres=[waldres RF.resid(1+xx:end-nleads+xx,:)];
end
olsfull=OLSmontecarlo(cleaninst,waldres,const,10,10);
% Ljung Box test on predicted value
cleaninst_hat=olsfull.e'*[ones(1,size(waldres,1)); waldres'];
[~,p,LBQ]=lbqtest(cleaninst_hat,'Lags',parlags);
Q.BICwold=olsfull.BIC;
Q.LBQ=LBQ;
Q.p=p;


end