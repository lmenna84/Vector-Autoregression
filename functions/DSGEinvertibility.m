clear

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Leeper model 

alf=0.36;
thet=0.2673;
tauss=0.25;
kappa=(1-thet)*tauss/(1-tauss);
gam=0.7;

syms k tau T add1 add2 add3 d
syms k_f tau_f T_f add1_f add2_f add3_f d_f
syms k_l tau_l T_l add1_l add2_l add3_l d_l
syms a ut ud
    
Y=[k tau T add1 add2 add3 d];
Y_f=[k_f tau_f T_f add1_f add2_f add3_f d_f];
Y_l=[k_l tau_l T_l add1_l add2_l add3_l d_l];

% set tau=add3 for non invertible and tau=add1 for invertible model
EQNlin=[k-alf*k_l-a+kappa*T;...
    T-tau_f-thet*T_f;...
    add1-ut;...
    add2-add1_l;...
    add3-add2_l;...
    d-gam*d_l-ud;
    %tau-add3-d;];
    tau-add1-d;];

addpath C:\Users\C14569\Desktop\my_matlab_functions\DSGE

Q=chomoreno(EQNlin,Y,[a ut ud],[1 1 1],[],1,[],[],1)
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% simulate data

%for xx=1:1000
sim_length=1000;
init=zeros(7,1);
shock=randn(2,sim_length);
% set shockdem to zero for recoverability
shockdem=randn(1,sim_length);
Y=zeros(7,sim_length);
for ii=1:sim_length
    if ii==1
        Y(:,ii)=Q.V1*init+Q.V2*[shock(:,ii); shockdem(:,ii)];
    else Y(:,ii)=Q.V1*Y(:,ii-1)+Q.V2*[shock(:,ii); shockdem(:,ii)];
    end
end
proxyinit=0;
for ii=1:sim_length
    if ii==1
        proxy(1,ii)=shock(2,ii)+randn(1,1)+0.5*proxyinit+0.4*init(1,1)-0.6*init(2,1);
    else proxy(1,ii)=shock(2,ii)+randn(1,1)+0.5*proxy(1,ii-1)+0.4*Y(1,ii-1)-0.6*Y(2,ii-1);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Invertibility test


invtest=invertibility(Y(1:2,:)',proxy',2,4,1,1)

rectest=recoverability(Y(1:2,:)',proxy',2,4,1,1)

% i(xx)=invtest.p;
% r(xx)=rectest.p;

%end

% count1=0;
% count2=0;
% for xx=1:1000
%     if r(xx)>0.05
%         count1=count1+1;
%     end
%     if i(xx)>0.05
%         count2=count2+1;
%     end
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% IV-VAR estimate

EQ=ivSVAR_blockbootstrap(Y(2,:)',Y(1,:)',proxy',2,{'tax' 'capital'},10,[],1)

close all

figure(1)
subplot(1,2,1)
plot(1:9,EQ.tax_tax(1,2:end),'--k',...
    1:9,EQ.tax_tax(2,2:end),'-k',...
    1:9,EQ.tax_tax(3,2:end),'--k',...
    1:9,Q.irf.ut(2,2:10),'-r')
title('tax')
subplot(1,2,2)
plot(1:9,EQ.capital_tax(1,2:end),'--k',...
    1:9,EQ.capital_tax(2,2:end),'-k',...
    1:9,EQ.capital_tax(3,2:end),'--k',...
    1:9,Q.irf.ut(1,2:10),'-r')
title('capital')

corr(EQ.struc',shock(2,3:end)')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% GIV-VAR estimate

EQ=givSVAR(Y(2,:)',Y(1,:)',proxy',2,4,2,{'tax' 'capital'},10,[],1,[],[],[],1)

close all

figure(1)
subplot(1,2,1)
plot(1:9,EQ.abs_tax(1,2:end),'--k',...
    1:9,EQ.abs_tax(2,2:end),'-k',...
    1:9,EQ.abs_tax(3,2:end),'--k',...
    1:9,Q.irf.ut(2,2:10),'-r')
title('tax')
subplot(1,2,2)
plot(1:9,EQ.abs_capital(1,2:end),'--k',...
    1:9,EQ.abs_capital(2,2:end),'-k',...
    1:9,EQ.abs_capital(3,2:end),'--k',...
    1:9,Q.irf.ut(1,2:10),'-r')
title('capital')

corr(EQ.struct',shock(2,3:end-4)')

figure(1)
subplot(1,2,1)
plot(1:9,EQ.upbound_tax(1,2:end),'--k',...
    1:9,EQ.upbound_tax(2,2:end),'-k',...
    1:9,EQ.upbound_tax(3,2:end),'--k',...
    1:9,Q.irf.ut(2,2:10),'-r')
title('tax')
subplot(1,2,2)
plot(1:9,EQ.upbound_capital(1,2:end),'--k',...
    1:9,EQ.upbound_capital(2,2:end),'-k',...
    1:9,EQ.upbound_capital(3,2:end),'--k',...
    1:9,Q.irf.ut(1,2:10),'-r')
title('capital')

figure(1)
subplot(1,2,1)
plot(1:9,EQ.lowbound_tax(1,2:end),'--k',...
    1:9,EQ.lowbound_tax(2,2:end),'-k',...
    1:9,EQ.lowbound_tax(3,2:end),'--k',...
    1:9,Q.irf.ut(2,2:10),'-r')
title('tax')
subplot(1,2,2)
plot(1:9,EQ.lowbound_capital(1,2:end),'--k',...
    1:9,EQ.lowbound_capital(2,2:end),'-k',...
    1:9,EQ.lowbound_capital(3,2:end),'--k',...
    1:9,Q.irf.ut(1,2:10),'-r')
title('capital')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Internal Instrument Estimate
IIE=srSVAR([proxy' Y(1:2,:)'],2,{'proxy' 'capital' 'tax'},10,[],1)

close all
% Renormalize in relative terms with respect to period 3 if non-invertible
% model
IIE.tax_proxy=IIE.tax_proxy./IIE.tax_proxy(:,4);
IIE.capital_proxy=IIE.capital_proxy./IIE.tax_proxy(:,4);

figure(1)
subplot(1,2,1)
plot(1:9,IIE.tax_proxy(1,2:end),'--k',...
    1:9,IIE.tax_proxy(2,2:end),'-k',...
    1:9,IIE.tax_proxy(3,2:end),'--k',...
    1:9,Q.irf.ut(2,2:10),'-r')
title('tax')
subplot(1,2,2)
plot(1:9,IIE.capital_proxy(1,2:end),'--k',...
    1:9,IIE.capital_proxy(2,2:end),'-k',...
    1:9,IIE.capital_proxy(3,2:end),'--k',...
    1:9,Q.irf.ut(1,2:10),'-r')
title('capital')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% IVLP estimate

lp=ivLP(Y(2,:)',Y(1,:)',proxy',[],9,2,[],[],{'tax' 'capital'},1,0)

% Renormalize in relative terms with respect to period 3 if non-invertible
% model
irftax=lp.tax./lp.tax(:,3);
irfcap=lp.capital./lp.tax(:,3);
close all

figure(1)
subplot(1,2,1)
plot(1:9,irftax(1,:),'--k',...
    1:9,irftax(2,:),'-k',...
    1:9,irftax(3,:),'--k',...
    1:9,Q.irf.ut(2,2:10),'-r')
title('tax')
subplot(1,2,2)
plot(1:9,irfcap(1,:),'--k',...
    1:9,irfcap(2,:),'-k',...
    1:9,irfcap(3,:),'--k',...
    1:9,Q.irf.ut(1,2:10),'-r')
title('capital')



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Univariate model (useful to understand)

% Thoretical model is: y(t)=0.5*y(t-1)+eps(t-1), where eps is the structural
% shock. Model is not invertible, because shock anticipated. For non
% recoverability, I can add another contemporaneous strctural shock, like
% y(t)=0.5*y(t-1)+eps(t-1)+h(t)
clear
% true irf
shock=[1 zeros(1,9)];
for ii=1:10
    if ii==1
        irf(1,ii)=0;
    else irf(1,ii)=0.5*irf(1,ii-1)+shock(1,ii-1);
    end
end
for xx=1:1000
% simulated data
dim=1000000;
eps=randn(1,dim);
% set h to zero for recoverability to hold
h=randn(1,dim+1)*0;
y=zeros(1,dim);
y(1)=y(1)+h(1);
for ii=1:dim
    y(1,ii+1)=0.5*y(1,ii)+eps(1,ii)+h(ii+1);
end
y(end)=[];
% standard choleski is wrong
% SR=srSVAR([y'],1,{'y'},10,100,1)
% close all
% irfchol=SR.y_y(:,2:end);
% figure(1)
% plot(1:9,irfchol(1,:),'--k',...
%     1:9,irfchol(2,:),'-k',...
%     1:9,irfchol(3,:),'--k',...
%     1:9,irf(1:9),'-r')

% generate simple proxy variable
z=eps+randn(1,dim);
% test invertibility and recoverability with proxy
I=invertibility(y',z',1,4,1);
R=recoverability(y',z',1,4,1);
i(xx)=I.p;
r(xx)=R.p;
RF=reducedformVAR(y',1,1);
e=OLS(z(2:end-1),[RF.resid(1:end-1) RF.resid(2:end)],1)
pred=e'*[ones(size(RF.resid,1)-1,1) RF.resid(1:end-1) RF.resid(2:end)]';
end

count1=0;
count2=0;
for xx=1:1000
    if r(xx)>0.1
        count1=count1+1;
    end
    if i(xx)>0.1
        count2=count2+1;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
clear

siz=100000;
eps=randn(3,siz);
nu=randn(1,siz);
Y=zeros(3,siz+1);
A=[0.2 0.4 0.3;0.1 0.6 -0.3;0.3 0.6 -0.5];
P=[0.5 -0.6 0.5;0.2 0.9 0.1;-0.9 0.6 0.2];
for ii=1:siz
    Y(:,ii+1)=A*Y(:,ii)+P*eps(:,ii);
end
Y=Y(:,2:end);
proxy=2*eps(1,:)+nu;
irf=zeros(3,11);
w=zeros(3,11);
w(1,1)=1;
for ii=1:11
    irf(:,ii+1)=A*irf(:,ii)+P*w(:,ii);
end

inv=invertibility(Y(1:2,:)',proxy',1,1,0,1)
rec=recoverability(Y(1:2,:)',proxy',1,1,0,1)



EQ=ivSVAR_blockbootstrap(Y(1,:)',Y(2,:)',proxy',1,{'tax' 'cap'},10,100,1)

