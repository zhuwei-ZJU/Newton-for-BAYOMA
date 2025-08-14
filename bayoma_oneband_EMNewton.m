function out = bayoma_oneband_EMNewton(in)
%=========================================
% Data preparation
%=========================================
fs = in.fs;
f1f2 = in.f1f2;
% detrend
in.tdata = detrend(in.tdata);
% scaling: eliminate the amptitude difference among modal parameters
in.scale = 1/max(std(in.tdata));
in.tdata = in.tdata*in.scale;
% scaled fft
[nt,n] = size(in.tdata);   % [number of time domian points, number of dofs]
nf0 = floor((nt+1)/2); % no. of ffts up to Nyquist, will change later
ff0 = (0:nf0-1).'*fs/nt; % (nf0,1), ordinates of freq.
F = sqrt(2/fs/nt)*fft(in.tdata);    % scaled FFT - two sided
F = F(2:nf0,:);
ff = ff0(2:nf0); % (:,1)
I1I2 = round(f1f2*nt/fs);
II = I1I2(1):I1I2(2);
F = F(II,:); % (nf0,n,nseg)
ff = ff(II); % (nf0,1)
nf = length(ff(:));   % number of fft points
% clear tdata to save space
in = rmfield(in,'tdata');
%=========================================
% Initial setting
%=========================================
f0 = in.f0;
z0 = 0.01;
m = length(f0);   % number of modes
% initialize f and z
f = f0.';
z = z0*ones(m,1);
% initialize phi
sv0 = zeros(1,m);
phi0 = zeros(n,m);
for ii=1:m
    I0 = round((f0(ii)-ff(1))*nt/fs)+1; % index corresponding to initial guess freq.
    dn0 = round(0.005*nt*f0(ii)/fs); % half bandwidth round f0 for taking average
    II = find((1:nf)<=I0+dn0 & (1:nf)>=I0-dn0);
    DD = real(F(II,:).'*conj(F(II,:))); % (n,n)
    DD = DD/length(II);
    [vtmp,stmp] = svd(DD);
    phi0(:,ii) = vtmp(:,1);
    sv0(ii) = stmp(ii,ii);
end
% arrange f & sv0 in descending order of magnitude, to be consistent w/ BA1
[~,I] = sort(sv0,'descend');
f = f(I);
phi0 = phi0(:,I);

phi00 = phi0;

% calculate A0 & determine the modeshape space
A0 = real(F.'*conj(F)); % (n,n)
[BA,dA] = svd(A0);
dA = diag(dA); % (n,1), singular values in descending order
mp = min(m,n);
dA1 = dA(1:mp);
if m>1 && max(abs(calMAC(phi0)))>0.9
    phi0 = BA(:,1:m);
end
% project phi0 onto MSS to get a refined guess phi
BA1 = BA(:,1:mp); % (n,mp)
alph = (BA1.'*BA1)\(BA1.'*phi0); % (m,m)
alph = normc(alph); % all col. have unit norm
phi = BA1*alph; % (n,m)

% initialize S
betak = f./ff.';
ihik = 1-betak.^2 - 1i*2*z.*betak; % hik^-1
mf0i = F*phi/(phi'*phi).*ihik.';    % (hik.^-1*Qk).'
S = (mf0i'*mf0i).'/nf;
% initialize Se
traceA0 = trace(A0);
ResA0 = (traceA0 - sum(dA1)); % residual sum of eigenvaleus of A0
if n == mp % note that n>=mp always
    Se1 = eigs(real(F(1:10,:).'*conj(F(1:10,:)))/10,m);
    Se = min(Se1);
else
    Se = ResA0/nf/(n-mp);
end
% Optimization
%=========================================
% Part I: EM (f,z updated by Newton)
%=========================================
nIterEM = in.maxEMIter;
tEM = 1;
tfz = 1;
xxtEM = mpmat2vec(f,z,phi,S,Se);
Set = Se;
nIterNewton = in.maxNewtonIter;
tNewton = 1;
ntheta = (m+1)^2+m*n;
XX = zeros(ntheta,nIterEM+nIterNewton+1);
XX(:,1) = xxtEM;
GRAD = zeros(nIterEM+nIterNewton+1,ntheta);
CM = zeros(ntheta,ntheta,nIterNewton);
steplen = zeros(nIterNewton,1);
F3(:,1,:) = F.';   % [n,1,nf]
disp('EM algorithm begins...');
fprintf(['IterEM.\tfreq (Hz) ',repmat(' ',1,10*(m-1))]);
fprintf(['damping   ',repmat(' ',1,10*(m-1))]);
fprintf('Se        ');
fprintf(['Sii       ',repmat(' ',1,10*(m-1))]);
fprintf('\n');
tic;
while tEM <= nIterEM
    fprintf('%-3i\t',tEM);
    fprintf('%-6.2e ',[f(:).',z(:).',Se/in.scale^2,diag(S).'/in.scale^2]);
    fprintf('\n');
    % E step
    [Mu,~,Mu2,~] = EYitak12(f,z,phi,S,Se,ff,F);
    sumReMuF3 = sum(real(pagemtimes(Mu,'none',F3,'ctranspose')),3);   % [m,n]
    sumReMu2 = sum(real(Mu2),3);   % [m,m]
    sumF3F3 = sum(pagemtimes(F3,'ctranspose',F3,'none'),3);   % scalar
    beta(:,1,:) = f./ff.';   % [m,1,nf]
    ih = 1 - beta.^2 - 1i*2*beta.*z;   % [m,1,nf]
    sumihMu2ihT = sum(ih.*Mu2.*pagectranspose(ih),3);   % [m,m]
    % M step
    phi = sumReMuF3.'/sumReMu2;
    Se = 1/n/nf*(sumF3F3-2*trace(phi*sumReMuF3)+trace(phi*sumReMu2*phi.'));
    S = sumihMu2ihT/nf;
    % Rejection to bypass mode shape norm constraints
    rnorm = sqrt(sum(phi.^2));
    phi = phi./rnorm;
    S = diag(rnorm)*S*diag(rnorm);
    if tfz == 1
        QfunHt = calQfunH([f;z],phi,S,Se,ff,F);
    end
    if tfz <= in.maxfzIter
        %=====================================
        % Newton's method to optimize f and z
        %=====================================
        [dQ2,ddQ2] = caldQfunH(f,z,phi,S,Se,ff,F);
        fz = [f;z] - ddQ2\dQ2.';
        ftp1 = fz(1:m);
        ztp1 = fz(m+1:end);
        QfunHtp1 = calQfunH([ftp1;ztp1],phi,S,Se,ff,F);
        if QfunHtp1>QfunHt || any(f<f1f2(1))||any(f>f1f2(2))||any(z<0)||any(z>1)
            if tfz > 1
                f = XX(1:m,tfz-1);
                z = XX(m+1:2*m,tfz-1);
            end
            tfz = in.maxfzIter+1;
        else
            f = ftp1;
            z = ztp1;
            QfunHt = QfunHtp1;
            tfz = tfz+1;
        end
    end
    XX(:,tEM+1) = mpmat2vec(f,z,phi,S,Se);
    % Covergence test
    if abs(Se-Set)/Set < in.tol_cvg && tEM >= in.minEMIter
        fprintf(['EM part stops as the change of Se is less than ' ...
            'a relative tolerance of %5.3e\n'],in.tol_cvg);
        tEM = tEM + 1;
        break;
    else
        Set = Se;
    end
    tEM = tEM + 1;
end
time_EM = toc;
if tEM > nIterEM
    fprintf('EM part stops as it meets a maximum iteration of %5.3e\n',in.maxEMIter);
end
disp(['Elapsed time is ',num2str(time_EM,8),' seconds.']);
%=========================================
% Part II: Newton's method
%=========================================
disp('Newton''s method begins...');
fprintf(['IterNewton.\tfreq (Hz) ',repmat(' ',1,10*(m-1))]);
fprintf(['damping   ',repmat(' ',1,10*(m-1))]);
fprintf('Se        ');
fprintf(['Sii       ',repmat(' ',1,10*(m-1))]);
fprintf('\n');
xxtNewton = XX(:,tEM);
Lt = calNLLF(ff,F,f,z,phi,S,Se);
tic;
while tNewton <= nIterNewton
    fprintf('%-3i\t',tNewton);
    fprintf('%-6.2e ',[f(:).',z(:).',Se/in.scale^2,diag(S).'/in.scale^2]);
    fprintf('\n');
    Ltp1 = 1e10;
    [g,H] = NLLFHess_scalarSe(f,z,phi,S,Se,ff,F);
    U = cstgradnull(phi);
    Hp = U.'*H*U;   % Hessian account for the norm constraints
    [VHp,DHp] = eig(Hp);
    DHp = real(diag(DHp));
    VHp = real(VHp);
    if any(DHp<=0)
        iDHp = 1./abs(DHp);
        CM(:,:,tNewton) = U*VHp*diag(iDHp)*VHp.'*U.';
    else
        CM(:,:,tNewton) = U/(U.'*H*U)*U.';
    end
    dk = -CM(:,:,tNewton)*g.';
    dt = 1;
    while Ltp1>Lt || any(f<f1f2(1))||any(f>f1f2(2))||any(z<0)||any(z>1)...
            ||Se<0||III
        xxNewton = xxtNewton+dt*dk;
        [f,z,phi,S,Se] = mpvec2mat(xxNewton,n,m);
        % Rejection to bypass mode shape norm constraints
        rnorm = sqrt(sum(phi.^2));
        phi = phi./rnorm;
        S = diag(rnorm)*S*diag(rnorm);
        Ltp1 = calNLLF(ff,F,f,z,phi,S,Se);
        if any(eig(S)<0)
            III = 1;
        else
            III = any(vec(abs(S./sqrt(diag(S)*diag(S).')))>1);
        end
        dt = dt/2;
        if dt <= in.steplentol
            break;
        end
    end
    steplen(tNewton) = dt*2;
    XX(:,tEM+tNewton) = xxNewton;
    % Covergence test
    INewton = find(xxtNewton);
    if all(abs(xxNewton(INewton)-xxtNewton(INewton))./xxtNewton(INewton) < in.tol_cvg) &&...
        tNewton >= in.minNewtonIter
        fprintf(['Newton part stops as the parameter changes are less than' ...
            ' a relative tolerance of %5.3e\n'],in.tol_cvg);
        tNewton = tNewton+1;
        break;
    else
        xxtNewton = xxNewton;
        Lt = Ltp1;
    end
    tNewton = tNewton+1;
end
time_Newton = toc;
if tNewton > nIterNewton
    fprintf('Newton part stops as it meets a maximum iteration of %5.3e\n',in.maxNewtonIter);
end
disp(['Elapsed time is ',num2str(time_Newton,8),' seconds.']);
%=========================================
% Set output
%=========================================
XX(:,tEM+tNewton:end) = [];
GRAD(tEM+tNewton:end,:) = [];
L = zeros(tEM+tNewton-1,1);
for ii = 1:tEM+tNewton-1
    [f00,z00,phi00,S00,Se00] = mpvec2mat(XX(:,ii),n,m);
    [L(ii),GRAD(ii,:)] = calNLLF(ff,F,f00,z00,phi00,S00,Se00);
end
% GRAD(:,end-m^2:end) = GRAD(:,end-m^2:end)/in.scale^2;
XX(end-m^2:end,:) = XX(end-m^2:end,:)/in.scale^2;
CM(:,:,tNewton:end) = [];
steplen(tNewton:end) = [];
% Optimum of the modal parameters
[out.f,IIf] = sort(f.');
out.z = z(IIf).';
out.phi = phi(:,IIf);
out.S{1} = S(IIf,IIf)/in.scale^2;
out.Sii = real(diag(out.S{1})).';
out.Se = Se/in.scale^2*ones(1,m);
out.rms = sqrt(pi/4*diag(out.S{1}).'.*out.f./out.z);
out.sn = out.Sii/4./out.z.^2./out.Se;
% Posterior COV of the modal parameters
PCM = real(CM(:,:,end));
diagPCM = diag(PCM).';
out.coefv.f = sqrt(diagPCM(1:m))./f.';
out.coefv.z = sqrt(diagPCM(m+1:2*m))./z';
out.coefv.Sii = sqrt(diagPCM(2*m+m*n+1:2*m+m*n+m))./real(diag(S)).';
out.coefv.Se = sqrt(diagPCM(end))./Se*ones(1,m);
out.coefv.phi = sqrt(sum(reshape(diagPCM(2*m+1:2*m+m*n)',n,m)));
% resort according to the value of the natural frequency
out.coefv.f = out.coefv.f(IIf);
out.coefv.z = out.coefv.z(IIf);
out.coefv.Sii = out.coefv.Sii(IIf);
out.coefv.phi = out.coefv.phi(IIf);
out.coefv.rms = out.coefv.f.^2 + out.coefv.z.^2 + out.coefv.Sii.^2 ...
    -2*diag(PCM(1:m,m+1:2*m)).' + 2*diag(PCM(1:m,2*m+1:3*m)).' ...
    -2*diag(PCM(m+1:2*m,2*m+1:3*m)).';   % (1,m), c.o.v.^2 of rms^2
% Iterated parameters, function values, gradients and covriance matrices
out.XX{1} = real(XX);
out.loglik{1} = -(L - 2*n*nf*log(in.scale));
out.grad{1} = GRAD;
out.normgrad{1} = vecnorm(GRAD.').';
out.CM{1} = CM;
% Iterated step length
out.steplen{1} = steplen;
% Number of iterations and consuming time
out.nEMIter = tEM-1;
out.nNewtonIter = tNewton-1;
out.time_EM = time_EM;
out.time_Newton = time_Newton;
out.time_req = time_EM+time_Newton;
end