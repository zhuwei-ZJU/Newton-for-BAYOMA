function xx = mpmat2vec(f,z,PHI,S,Se)
xx = [f;z;vec(PHI);real(diag(S));real(vec(S,'hh'));imag(vec(S,'hh'));Se];
end