%% Calculate some frequently used 1st and 2nd derivative terms of Q function
% Note only the A13,14,15,16 and B18,19,20,24,25 are different from ABterms
%========================================================================
% 220924-Firstly written by Wei at ZJUI
% 221108-Use 'pagemtimes' function to get ABterms at all frequency points
% 230523-More condensed form for Ak terms
%========================================================================
% Input
%========================================================================
% f: Natural frequency [m,1]
% z: Damping ratio [m,1]
% PHI: Mode shape [n,m]
% S: Modal force PSD [m,m]
% Se: Prediction error: scalar
% ff: Frequency, [nf,1]
% F: FFT of measured data, [nf,n]
% Mu,Mu2: The 1st and 2nd central mement of the latent variable
%========================================================================
% Output
%========================================================================
% A: terms not depending on the latent variable [ntheta,(m+1)^2]
% HQ: Hessian matrix of the Q-function
%========================================================================
function [A,HQ] = calAHQ(f,z,PHI,S,Se,ff,F,Mu,Mu2)
[n,m] = size(PHI);   % [number of dof, number of modes]
ntheta = (m+1)^2+m*n;   % number of parameters
nf = length(ff);   % number of fft points
beta = f./ff.';   % [m,nf]
h = 1./(1 - beta.^2 - 1i*2*beta.*z);  ih = 1./h;    % [m,nf]
D = 1./((1-beta.^2).^2+4*z.^2.*beta.^2);   D2 = D.^2;   % [m,nf]
ff2 = ff.^2; ff4 = ff.^4; ff6 = ff.^6;   % [nf,1]
f2 = f.^2; f3 = f.^3; f4 = f.^4;   % [m,1]
z2 = z.^2;

f_ff = f./ff.';   f_ff2 = f./ff2.';   z_ff = z./ff.';   % [m,nf]
dhdf(:,1,:) = 2*f_ff2 + 2i*z_ff;   % -d(diag(inv(hk)))/df   [m,1,nf]
dhdf = pagediag(dhdf);   % [m,m,nf]
dhdz(:,1,:) = 2i*f_ff;   % -d(diag(inv(hk)))/dz   [m,1,nf]
dhdz = pagediag(dhdz);   % [m,m,nf]
dhpdf = conj(dhdf);   % -d(diag(inv(hk')))/df   [m,m,nf]
iS = inv(S);   veciS = vec(iS.');   iSiS = kron(iS.',iS);
ih3(:,1,:) = ih;   % [m,1,nf]
iSdpihp = iS.'.*pagectranspose(ih3);   iSih = iS.*pagetranspose(ih3);    % [m,m,nf]
Rm = full(vec2diag(m));   Kmm = com_mat(m,m);  Im = eye(m);  In = eye(n);
iSe = 1/Se;   iSe2 = 1/Se^2;   iSe3 = 1/Se^3;
ImiSePHI = kron(Im,iSe*PHI);
vecPHIPHI = vec(PHI.'*PHI).';   % for scalar Se
%========================================================================
% First derivative terms that does not depend on the latent variable
%========================================================================
% f
A1 = pagemtimes2(Rm,pagekron(dhdf,iSdpihp),Kmm);   % [m,m^2,nf]
A2 = pagemtimes(conj(A1),Kmm);   % [m,m^2,nf]
A3(:,1,:) = -4*D.*f./ff2.' + 4*D.*f3./ff4.' + 8*D.*f.*z2./ff2.';   % [m,1,nf]
% z
A4 = pagemtimes2(Rm,pagekron(dhdz,iSdpihp),Kmm);   % [m,m^2,nf]
A5 = pagemtimes(conj(A4),Kmm);   % [m,m^2,nf]
A6(:,1,:) = 8*D.*f2.*z./ff2.';   % [m,1,nf]
% PHI
F3(:,1,:) = F.';   % [n,1,nf]
A7(:,1:m,:) = iSe*pagekron(Im,F3);   % [m*n,m,nf]
A8 = conj(A7);   % [m*n,1,nf]
a1 = ones(m*n,m^2,nf);
A9 = -ImiSePHI.*a1;   % [m*n,m^2]
A10 = -ImiSePHI*Kmm.*a1;   % [m*n,m^2]
% S
A11 = pagemtimes(Kmm,pagekron(iSdpihp,iSih));   % [m^2,m^2,nf]
a2 = ones(m^2,1,nf);
A12 = -veciS.*a2;   % [m^2,1]
% Se
A13(1,:,:) = -iSe2*PHI.'*F.';   % [1,m,nf]
A14 = conj(A13);   % [1,m,nf]
A15 = iSe2*vecPHIPHI.*pagetranspose(a2);   % [1,m^2]
diagFF = diag(F*F');
A16(1,1,:) = -n*iSe + iSe2*diagFF;   % [1,1,nf]
Omm = zeros(m,m,nf);   Omn1 = zeros(m*n,1,nf);   Om2m = zeros(m^2,m,nf);
A = [A1+A2,Omm,Omm,A3
    A4+A5,Omm,Omm,A6
    A9+A10,A8,A7,Omn1
    A11,Om2m,Om2m,A12
    A15,A14,A13,A16];   % [ntheta,(m+1)^2,nf]
%========================================================================
% Second derivative terms
%========================================================================
HQ = zeros(ntheta,ntheta,nf);
If = 1:m;   Iz = m+1:2*m;   IPHI = 2*m+1:2*m+m*n;
IS = 2*m+m*n+1:2*m+m*n+m^2; ISe = ntheta;
% ff
Mu2dp = pagetranspose(Mu2);
fff2(1,:,:) = ff2.';
B1 = 2*pagemtimes(iSdpihp,Mu2dp).*Im./fff2...
    -pagemtimes(pagemtimes(dhdf,Mu2).*iS.',dhpdf);   % [m,m,nf]
B2 = conj(B1);   % [m,m,nf]
B3(:,1,:) = D2.*(-16*f2./ff4.' + 32*f4./ff6.' + 64*z2.*f2./ff4.' - 16*f.^6./ff.^8.' ...
    - 64*z2.*f4./ff6.' - 64*z.^4.*f2./ff4.') + D.*(-4./ff2.'+12*f2./ff4.'+8*z2./ff2.');
B3 = pagediag(B3);   % [m,m,nf]
% fz
fff(1,:,:) = ff.';
B4 = 2i*pagemtimes(iSdpihp,Mu2dp)./fff.*Im + ...
    pagemtimes(pagemtimes(dhdf,Mu2).*iS.',dhdz);   % [m,m,nf]
B5 = conj(B4);    % [m,m,nf]
B6(:,1,:) = 32*D2.*f3.*z./ff4.' - 32*D2.*f.^5.*z./ff6.'   ...
    -64*D2.*f3.*z.^3./ff4.' + 16*D.*f.*z./ff2.';  % [m,1,nf]
B6 = pagediag(B6);   % [m,m,nf]
% fs
Mu2ihp = Mu2.*pagectranspose(ih3);
B7 = -pagemtimes2(Rm,pagekron(Im,pagemtimes(dhdf,Mu2ihp)),iSiS); % [m,m^2,nf]
B8 = pagemtimes(conj(B7),Kmm);
% zz
B9(:,1,:) = -64*D2.*f4.*z2./ff4.' + 8*D.*f2./ff2.';   % [m,1,nf]
B9 = pagediag(B9);   % [m,m,nf]
B10 = pagemtimes(pagemtimes(dhdz,Mu2).*iS.',dhdz);   % [m,m,nf]
B11 = conj(B10);   % [m,m,nf]
% zS
B12 = -pagemtimes2(Rm,pagekron(Im,pagemtimes(dhdz,Mu2ihp)),iSiS);   % [m,m,nf]
B13 = pagemtimes(conj(B12),Kmm);   % [m,m,nf]
% PHIPHI
B14 = -iSe*pagekron(Mu2dp,In);   % [mn,mn,nf]
B15 = conj(B14);   % [mn,mn,nf]
% PHISe
B16 = iSe2*pagevec(-pagemtimes(F3,'none',Mu,'ctranspose')- ...
    pagemtimes(conj(F3),'none',Mu,'transpose')+pagemtimes(PHI,2*real(Mu2)));   % [mn,1,nf]
% SS
iSihMu2ihpiS = pagemtimes2(iSih,Mu2,iSih,'ctranspose');
B17 = -pagemtimes(Kmm,pagekron(pagetranspose(iSihMu2ihpiS),iS));   % [m^2,m^2,nf]
B18 = -pagemtimes(Kmm,pagekron(iS.',iSihMu2ihpiS));   % [m^2,m^2,nf]
B19 = Kmm*iSiS;   % [m^2,m^2]
% SeSe
B20(1,1,:) = n*iSe2 - 2*iSe3*diagFF;   % [1,1,nf]
MupPHIpF3 = pagemtimes2(Mu,'ctranspose',PHI,'transpose',F3);
B21 =  2*iSe3*(2*real(MupPHIpF3)-pageprodtr(PHI.'*PHI,Mu2));   % [1,1,nf]
HQ(If,If,:) = B1 + B2 + B3;
HQ(If,Iz,:) = B4 + B5 + B6;
HQ(If,IS,:) = B7 + B8;
HQ(Iz,Iz,:) = B9 + B10 + B11;
HQ(Iz,IS,:) = B12 + B13;
HQ(IPHI,IPHI,:) = B14 + B15;
HQ(IPHI,ISe,:) = B16;
HQ(IS,IS,:) = B17 + B18 + B19;
HQ(ISe,ISe,:) = B20 + B21;
HQ = HQ.*triu(ones(ntheta));
HQ = HQ + pagetranspose(HQ) - HQ.*eye(ntheta);
end