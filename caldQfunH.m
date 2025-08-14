function [dQfunH,ddQfunH] = caldQfunH(f,z,PHI,S,Se,ff,F)
m = length(f);
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
Nu2 = pagevec(Mu2);
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
iS = inv(S); Im = eye(m);
ih3(:,1,:) = ih;   % [m,1,nf]
iSdpihp = iS.'.*pagectranspose(ih3);   iSih = iS.*pagetranspose(ih3);    % [m,m,nf]
Rm = full(vec2diag(m));
% f
A1 = pagemtimes(Rm,pagekron(iSdpihp,dhdf));   % [m,m^2,nf]
A2 = pagemtimes(Rm,pagekron(Im,pagemtimes(dhpdf,iSih)));   % [m,m^2,nf]
A3(:,1,:) = -4*D.*f./ff2.' + 4*D.*f3./ff4.' + 8*D.*f.*z2./ff2.';   % [m,1,nf]
% z
A4 = pagemtimes(Rm,pagekron(iSdpihp,dhdz));   % [m,m^2,nf]
A5 = -pagemtimes2(Rm,pagekron(Im,pagemtimes(dhdz,iSih)));   % [m,m^2,nf]
A6(:,1,:) = 8*D.*f2.*z./ff2.';   % [m,1,nf]
gf = pagemtimes(A1,Nu2) + pagemtimes(A2,Nu2) + A3;   % [m,1,nf]
gz = pagemtimes(A4,Nu2) + pagemtimes(A5,Nu2) + A6;   % [m,1,nf]
dQfunH = [sum(gf,3);sum(gz,3)].';   % [1,m]
dQfunH = -real(dQfunH);
if nargout > 1   % hessian matrix required
    ddQfunH = zeros(2*m,2*m);
    % ff
    Mu2dp = pagetranspose(Mu2);
    B1 = -pagemtimes(pagemtimes(dhdf,Mu2).*iS.',dhpdf);   % [m,m,nf]
    B2 = -pagemtimes(pagemtimes(dhpdf,iS).*Mu2dp,dhdf);   % [m,m,nf]
    fff2(1,:,:) = ff2.';
    B3 = 2*pagemtimes(iSdpihp,Mu2dp).*Im./fff2;   % [m,m,nf]
    B4 = 2*pagemtimes(Mu2,'transpose',iSdpihp,'ctranspose').*Im./fff2;   % [m,m,nf]
    B5(:,1,:) = D2.*(-16*f2./ff4.'+16*f4./ff6.'+32*z2.*f2./ff4.')-4*D./ff2.'  ...
        + D2.*(16*f4./ff6.'-16*f.^6./ff.^8.'-32*z2.*f4./ff6.')+12*D.*f2./ff4.'   ...
        + D2.*(32*z2.*f2./ff4.'-32*z2.*f4./ff6.'-64*z.^4.*f2./ff4.')+8*D.*z2./ff2.';   % [m,1,nf]
    B5 = pagediag(B5);   % [m,m,nf]
    % fz
    B6 = pagemtimes(pagemtimes(dhdf,Mu2).*iS.',dhdz);   % [m,m,nf]
    fff(1,:,:) = ff.';
    B7 = 2i*pagemtimes(iSdpihp,Mu2dp)./fff.*Im;   % [m,m,nf]
    B8 = -pagemtimes(pagemtimes2(dhpdf,iS).*Mu2dp,dhdz);   % [m,m,nf]
    B9 = -2i*pagemtimes(Mu2dp,'none',iSih,'transpose').*Im./fff;   % [m,m,nf]
    B10(:,1,:) = 32*D2.*f3.*z./ff4.' - 32*D2.*f.^5.*z./ff6.'   ...
        -64*D2.*f3.*z.^3./ff4.' + 16*D.*f.*z./ff2.';  % [m,1,nf]
    B10 = pagediag(B10);   % [m,m,nf]
    % zz
    B13(:,1,:) = -64*D2.*f4.*z2./ff4.' + 8*D.*f2./ff2.';   % [m,1,nf]
    B13 = pagediag(B13);   % [m,m,nf]
    B14 = pagemtimes(pagemtimes(dhdz,Mu2).*iS.',dhdz);   % [m,m,nf]
    B15 = pagemtimes(pagemtimes(dhdz,iS).*Mu2dp,dhdz);   % [m,m,nf]
    ddQfunH(1:m,1:m) = sum(B1 + B2 + B3 + B4 + B5,3);
    ddQfunH(1:m,m+1:2*m) = sum(B6 + B7 + B8 + B9 + B10,3);
    ddQfunH(m+1:2*m,1:m) = ddQfunH(1:m,m+1:2*m).';
    ddQfunH(m+1:2*m,m+1:2*m) = sum(B13 + B14 + B15,3);
    ddQfunH = -real(ddQfunH);
end
end