function metastates = copyhmm(metastates,P,Pi,Dir2d_alpha,Dir_alpha)
    metastates.Pi = Pi; metastates.P = P;
    metastates.Dir_alpha = Dir_alpha; metastates.Dir2d_alpha = Dir2d_alpha;
end