% This is the toy example of RKPCA in the following paper:
% "Exactly Robust Kernel Principal Component % Analysis" J Fan, TWS Chow. IEEE TNNLS 2019
clc
clear all
warning off
K=1;% number of subspaces
m=20;
n0=100;
r=2;
for u=1:1
    X=[]
    for k=1:K
%         X=[X randn(m,r)*randn(r,n0)]; % multiple linear subspaces
          Z=unifrnd(-1,1,r,n0);
          X=[X randn(m,r)*Z+0.5*randn(m,r)*Z.^2+0.5*randn(m,r)*Z.^3]; % multiple nonlinear subspaces
    end
%%
[m,n]=size(X);
e=randn(1,m*n);
noise_density=0.3;% proportion of corrupted entries
e(randperm(m*n,ceil(m*n*(1-noise_density))))=0;
E=reshape(e,size(X));
Xn=X+E;% sparse corruption
%% RPCA
par=[0.1:0.05:1];% lambda
for i=1:length(par)
    [X_rpca{i},E_t]=inexact_alm_rpca(Xn,par(i),1e-8,500);
    e_temp_rpca(i)=norm(X-X_rpca{i},'fro')/norm(X,'fro');
end
[~,idx]=min(e_temp_rpca);
Xr{1}=X_rpca{idx};
%% RKPCA
ker.type='rbf';
ker.par=[0];
par=[0.1:0.05:0.5];% lambda
options.p=1;
%options.r_svd=1;
for i=1:length(par)
    [X_rkpca{i},E_t,f]=RKPCA_PLMAdSS(Xn,ker,par(i),options);
    e_temp_rkpca(i)=norm(X-X_rkpca{i},'fro')/norm(X,'fro');
end
[~,idx]=min(e_temp_rkpca);
Xr{2}=X_rkpca{idx};
%% compute recovery error
for k=1:length(Xr)
    RE(u,k)=norm(X-Xr{k},'fro')/norm(X,'fro');
end
end
RE_mean=mean(RE)
