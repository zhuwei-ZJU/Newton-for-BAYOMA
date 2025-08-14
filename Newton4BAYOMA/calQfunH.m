function Q2 = calQfunH(fz,PHI,S,Se,ff,F)
%==========================================
% Input
%==========================================
% fz: [2m,1]
% PHI: [m,n]
% S: [m,m]
% Se: scalar
% ff: Frequency, [nf,1]
% F: FFT of measured data, [nf,n]
%==========================================
% Output
%==========================================
% Q2: Q-function value related to f and z
%==========================================
m = length(fz)/2;
f = fz(1:m);
z = fz(m+1:end);
beta(:,1,:) = f./ff.';   % [m,1,nf]
h = 1./(1 - beta.^2 - 1i*2*beta.*z);   % [m,1,nf]
H = h.*S.*pagectranspose(h);   % [m,m,nf]
iSe = diag(1./diag(Se));
P = pageinv(H) + PHI.'*iSe*PHI;   % [m,m,nf]
F3(:,1,:) = F.';   % [n,1,nf]
% Mu = pagemtimes(pagemldivide(P,PHI.')*iSe,F3);   % [m,1,nf]
Mu = pagemtimes2(pageinv(P),PHI.',iSe,F3);   % [m,1,nf]
Sigma = pageinv(P);   % [m,m,nf]
Mu2 = pagemtimes(Mu,'none',Mu,'ctranspose') + Sigma;   % [m,m,nf]
Q2 = sum(log(pagedet(H))) + trace(sum(pagemtimes(pageinv(H),Mu2),3));
Q2 = real(Q2);
end