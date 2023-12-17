import numpy as np

def Plus(*args):
    result = 0
    for i in args:
        result = result + i
    return result  

def Times(*args):
    result = 1
    for i in args:
        result = result * i
    return result  

def Rational(a,b):
    return a/b

def List(*args):
    return list(args)

Sin = np.sin
Cos = np.cos
Power = np.power


N = 200

x = np.linspace(-2,2,N)
y = np.linspace(-2,2,N)
z = np.linspace(-2,2,N)

X, Y, Z = np.meshgrid(x,y,z, indexing='ij')

theta = 6*X + 3*Y**2 + Z**3
phi   = X + Y + Z

n = np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta)
            ])
n = n.transpose((1,2,3,0))

splay_linear = Plus(Times(6, Y, Cos(theta), Sin(phi)), Times(-1, Plus(Times(3, \
Power(Z, 2)), Sin(phi)), Sin(theta)), Times(Cos(phi), Plus(Times(6, \
Cos(theta)), Sin(theta))))
                                                             
twist_linear = Times(Rational(1, 2), Plus(-1, Power(Cos(theta), 2), Times(12, \
Sin(phi)), Times(-1, Power(Sin(theta), 2)), Times(Sin(phi), \
Sin(Times(2, theta))), Times(Cos(phi), Plus(Times(-12, Y), \
Sin(Times(2, theta))))))
                                            
bend_linear = List(Plus(Times(Sin(theta), Plus(Times(Cos(theta), Plus(-6, Sin(phi), \
Times(6, Power(Sin(phi), 2)))), Times(Power(Sin(phi), 2), \
Sin(theta)))), Times(Cos(phi), Plus(Times(-3, Power(Z, 2), \
Power(Cos(theta), 2)), Times(-6, Y, Cos(theta), Sin(phi), \
Sin(theta)), Times(Sin(phi), Power(Sin(theta), 2))))), Plus(Times(-3, \
Power(Z, 2), Power(Cos(theta), 2), Sin(phi)), Times(-1, Cos(theta), \
Plus(Cos(phi), Times(6, Cos(phi), Sin(phi)), Times(6, Y, \
Power(Sin(phi), 2))), Sin(theta)), Times(-1, Cos(phi), Plus(Cos(phi), \
Sin(phi)), Power(Sin(theta), 2))), Times(3, Sin(theta), \
Plus(Times(Power(Z, 2), Cos(theta)), Times(2, Plus(Cos(phi), Times(Y, \
Sin(phi))), Sin(theta)))))
                                          
bend_linear = np.array(bend_linear).transpose((1,2,3,0))

levi = np.zeros((3,3,3))
levi[0,1,2], levi[1,2,0] ,levi[2,0,1] = 1, 1, 1
levi[1,0,2], levi[2,1,0] ,levi[0,2,1] = -1, -1, -1

width = 4
N = np.shape(n)[0]

Q = np.einsum('nmli, nmlj -> nmlij', n, n)
Q = Q - np.eye(3)/3

diffQ = np.zeros( (N, N, N, 3, 3, 3) )   # indexx, indexy, indexz, index_diff, index_Q1, indexQ2, 
diffQ[:, :, :, 0] = np.gradient(Q, axis=0) / ( width / (N-1) )
diffQ[:, :, :, 1] = np.gradient(Q, axis=1) / ( width / (N-1) )
diffQ[:, :, :, 2] = np.gradient(Q, axis=2) / ( width / (N-1) )


energy1 = np.einsum("ijkabc, ijkabc -> ijk", diffQ, diffQ)
energy2 = np.einsum("ijkaac, ijkbbc -> ijk", diffQ, diffQ)
energy3 = np.einsum("ijkab, ijkacd, ijkbcd -> ijk", Q, diffQ, diffQ)

splay =  - energy1  +  6 * energy2  - 3 * energy3
splay = splay / 6

twist_linear = np.einsum("abc, ijkad, ijkbcd -> ijk", levi, Q, diffQ)
twist = twist_linear**2

temp1 = np.einsum('nmlab, nmlbia -> nmli', Q, diffQ)
temp2 = np.einsum('nmlia, nmlbab -> nmli', Q, diffQ)
bend_vector = - 2 * temp1 - temp2
bend = np.sum(bend_vector**2, -1)





