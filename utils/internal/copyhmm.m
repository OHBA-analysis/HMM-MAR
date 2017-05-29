function states = copyhmm(states,P,Pi,Dir2d_alpha,Dir_alpha)
    states.Pi = Pi; states.P = P;
    states.Dir_alpha = Dir_alpha; states.Dir2d_alpha = Dir2d_alpha;
end