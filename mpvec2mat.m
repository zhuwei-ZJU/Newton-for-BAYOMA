function [f,z,PHI,S,Se] = mpvec2mat(xx,n,m)
mpC = mat2cell(xx,[m,m,m*n,m,m*(m-1)/2,m*(m-1)/2,1]);
PHI = zeros(n,m);
[f,z,PHI(:),diagS,ReSij,ImSij,Se] = deal(mpC{:});
vecS = vec2diag(m).'*diagS + dup_mat(m,'hh')*ReSij + 1i*dup_mat(m,'sk')*ImSij;
S = zeros(m,m);
S(:) = vecS;
S = (S+S')/2;
% f = xx(1:m).';
% z = xx(m+1:2*m).';
% PHI = reshape(xx(2*m+1:2*m+m*n),n,m);
% diagS = xx(2*m+m*n+1:2*m+m*n+m).';
% ReSij = xx(2*m+m*n+m+1:2*m+m*n+m+m*(m-1)/2).';
% ImSij = xx(2*m+m*n+m+m*(m-1)/2+1:2*m+m*n+m+m*(m-1)).';
% vecS = vec2diag(m).'*diagS + dup_mat(3,'hh')*ReSij + 1i*dup_mat(3,'sk')*ImSij;
% S = reshape(vecS,m,m);
% Se = xx(end);
end