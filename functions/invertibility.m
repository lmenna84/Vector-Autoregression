function Q=invertibility(data,iv,nlags,nleads,nlagsclean,const,controls)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Author: Lorenzo Menna %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Purpose: 
% Forni et al invertibility test. The null hipothesis is invertibility
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
% controls = NxZ matrix, Z number of control variables to include in
% instrumental variable regression
% -----------------------------------
% Returns:
% Q = structure
% Q.LBQ = Ljung Box statistics
% Q.p = p value of the test
% Q.F = F statistics
% Q.p = p value of the F test
% ------------------------------------

if nargin<7
    controls=[];
end
if nargin<6
    controls=[];
    const=0;
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
olsreduced=OLSmontecarlo(cleaninst,waldres(:,1:K),const,10,10);
% F test
sqresfull=olsfull.resid'*olsfull.resid;
sqresreduced=olsreduced.resid'*olsreduced.resid;
F=((sqresreduced-sqresfull)/(nleads*K))/(sqresfull/(size(cleaninst,1)-((nleads+1)*K+1))); 
p=1-fcdf(F,nleads*K,size(cleaninst,1)-((nleads+1)*K+1));
Q.BICwold=olsfull.BIC;
Q.F=F;
Q.p=p;


end