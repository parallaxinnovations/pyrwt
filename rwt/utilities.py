from __future__ import division
import numpy as np

def hardThreshold(y, thld):
    """
    hard thresholds the input signal y with the threshold value
    thld.

    Input:  
       y    : 1D or 2D signal to be thresholded
       thld : threshold value

    Output: 
       x : Hard thresholded output (x = (abs(y)>thld).*y)

  HERE'S AN EASY WAY TO RUN THE EXAMPLES:
  Cut-and-paste the example you want to run to a new file 
  called ex.m, for example. Delete out the  at the beginning 
  of each line in ex.m (Can use search-and-replace in your editor
  to replace it with a space). Type 'ex' in matlab and hit return.


    Example:
       y = makesig('WernerSorrows',8);
       thld = 1;
       x = HardTh(y,thld)
       x = 1.5545 5.3175 0 1.6956  -1.2678 0 1.7332 0

    See also: SoftTh

    """
    
    x = np.zeros_like(y)
    x[np.abs(y) > thld] = y
    
    return x


def softThreshold(y, thld):
    """
    soft thresholds the input signal y with the threshold value thld.

    Input:  
       y    : 1D or 2D signal to be thresholded
       thld : Threshold value

    Output: 
       x : Soft thresholded output (x = sign(y)(|y|-thld)_+)

  HERE'S AN EASY WAY TO RUN THE EXAMPLES:
  Cut-and-paste the example you want to run to a new file 
  called ex.m, for example. Delete out the  at the beginning 
  of each line in ex.m (Can use search-and-replace in your editor
  to replace it with a space). Type 'ex' in matlab and hit return.


    Example:
       y = makesig('Doppler',8);
       thld = 0.2;
       x = SoftTh(y,thld)
       x = 0 0 0 -0.0703 0 0.2001 0.0483 0 

    See also: HardTh

    Reference: 
       "De-noising via Soft-Thresholding" Tech. Rept. Statistics,
       Stanford, 1992. D.L. Donoho.
    """
    
    x = np.abs(y) - thld
    x[x<0] = 0
    x[y<0] = -x[y<0]


def makesig(SigName='AllSig', N=512):
    """
     Creates artificial test signal identical to the
     standard test signals proposed and used by D. Donoho and I. Johnstone
     in WaveLab (- a matlab toolbox developed by Donoho et al. the statistics
     department at Stanford University).

    Input:  SigName - Name of the desired signal (Default 'all')
                        'AllSig' (Returns a matrix with all the signals)
                        'HeaviSine'
                        'Bumps'
                        'Blocks'
                        'Doppler'
                        'Ramp'
                        'Cusp'
                        'Sing'
                        'HiSine'
                        'LoSine'
                        'LinChirp'
                        'TwoChirp'
                        'QuadChirp'
                        'MishMash'
                        'Werner Sorrows' (Heisenberg)
                        'Leopold' (Kronecker)
            N       - Length in samples of the desired signal (Default 512)

    Output: x   - vector/matrix of test signals
            N   - length of signal returned

    See also: 

    References:
            WaveLab can be accessed at
            www_url: http://playfair.stanford.edu/~wavelab/
            Also see various articles by D.L. Donoho et al. at
            web_url: http://playfair.stanford.edu/
    """

    t = np.linspace(1, N, N)/N
    x = np.zeros_like(t)
    
    if SigName in ('HeaviSine', 'AllSig'):
        y = 4 * np.sin(4*np.pi*t) - np.sign(t - 0.3) - sign(0.72 - t)
        
        x = np.vstack((x, y))
    
    if SigName in ('Bumps', 'AllSig'):
        pos = np.array([ .1, .13, .15, .23, .25, .40, .44, .65, .76, .78, .81])
        hgt = np.array([ 4,  5,   3,   4,  5,  4.2, 2.1, 4.3,  3.1, 5.1, 4.2])
        wth = np.array([.005, .005, .006, .01, .01, .03, .01, .01,  .005, .008, .005])
        y = np.zeros_like(t)
        for p, h, w in zip(pos, hgt, wth):
            y += h/(1 + np.abs((t - p)/w))**4

        x = np.vstack((x, y))
    
    if SigName in ('Blocks', 'AllSig'):
        pos = np.array([ .1, .13, .15, .23, .25, .40, .44, .65, .76, .78, .81])
        hgt = np.array([ 4,  -5,   3,   -4,  5,  -4.2, 2.1, 4.3,  -3.1, 2.1, -4.2])
        y = np.zeros_like(t)
        for p, h in zip(pos, hgt):
            y += (1 + np.abs(t - p))*h/2
            
        x = np.vstack((x, y))

    if SigName in ('Doppler', 'AllSig'):
        y = np.sqrt(t * (1-t)) * np.sin((2*np.pi*1.05) / (t+.05))
                
        x = np.vstack((x, y))

    if SigName in ('Ramp', 'AllSig'):
        y = t.copy()
        y[t >= .37] -= 1
                
        x = np.vstack((x, y))
    
    if SigName in ('Cusp', 'AllSig'):
        y = np.sqrt(np.abs(t - 0.37))
                
        x = np.vstack((x, y))

    if SigName in ('Sing', 'AllSig'):
        k = np.floor(N * .37)
        y = 1 / np.abs(t - (k+.5)/N)
                
        x = np.vstack((x, y))

"""
if(strcmp(SigName,'HiSine') | strcmp(SigName,'AllSig')),
  y = sin( pi * (N * .6902) .* t);
end;
x = [x;y];
y = [];
if(strcmp(SigName,'LoSine') | strcmp(SigName,'AllSig')),
  y = sin( pi * (N * .3333) .* t);
end;
x = [x;y];
y = [];
if(strcmp(SigName,'LinChirp') | strcmp(SigName,'AllSig')),
  y = sin(pi .* t .* ((N .* .125) .* t));
end;
x = [x;y];
y = [];
if(strcmp(SigName,'TwoChirp') | strcmp(SigName,'AllSig')),
  y = sin(pi .* t .* (N .* t)) + sin((pi/3) .* t .* (N .* t));
end;
x = [x;y];
y = [];
if(strcmp(SigName,'QuadChirp') | strcmp(SigName,'AllSig')),
  y = sin( (pi/3) .* t .* (N .* t.^2));
end;
x = [x;y];
y = [];
if(strcmp(SigName,'MishMash') | strcmp(SigName,'AllSig')),  
  % QuadChirp + LinChirp + HiSine
  y = sin( (pi/3) .* t .* (N .* t.^2)) ;
  y = y +  sin( pi * (N * .6902) .* t);
  y = y +  sin(pi .* t .* (N .* .125 .* t));
end;
x = [x;y];
y = [];
if(strcmp(SigName,'WernerSorrows') | strcmp(SigName,'AllSig')),
  y = sin( pi .* t .* (N/2 .* t.^2)) ;
  y = y +  sin( pi * (N * .6902) .* t);
  y = y +  sin(pi .* t .* (N .* t));
  pos = [ .1 .13 .15 .23 .25 .40 .44 .65  .76 .78 .81];
  hgt = [ 4  5   3   4  5  4.2 2.1 4.3  3.1 5.1 4.2];
  wth = [.005 .005 .006 .01 .01 .03 .01 .01  .005 .008 .005];
  for j =1:length(pos)
    y = y + hgt(j)./( 1 + abs((t - pos(j))./wth(j))).^4;
  end 
end;
x = [x;y];
y = [];
if(strcmp(SigName,'Leopold') | strcmp(SigName,'AllSig')),
  y = (t == floor(.37 * N)/N); 		% Kronecker
end;
x = [x;y];
y = [];

    """

    return (x, N)