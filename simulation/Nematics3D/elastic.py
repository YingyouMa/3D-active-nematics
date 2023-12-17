import numpy as np
from itertools import product
import time

levi = np.zeros((3,3,3))
levi[0,1,2], levi[1,2,0] ,levi[2,0,1] = 1, 1, 1
levi[1,0,2], levi[2,1,0] ,levi[0,2,1] = -1, -1, -1


def get_deform_n(n, width, if_print=True):

    N = np.shape(n)[0]
    diff = np.zeros( (N-2, N-2, N-2, 3, 3) )    # indexx, indexy, indexz, index_diff, index_n
    local = np.zeros((3,3,3,3))

    def find_diff(n, i,j,k):   

        local = n[i:i+3, j:j+3, k:k+3]

        def reorit(i,j,k):
            local[i,j,k] = np.sign(local[i,j,k]@local[0,1,1]) * local[i,j,k]

        reorit(2,1,1)
        reorit(1,0,1)
        reorit(1,2,1)
        reorit(1,1,0)
        reorit(1,1,2)

        diff[i,j,k,0,0] = local[2,1,1,0] - local[0,1,1,0]
        diff[i,j,k,0,1] = local[2,1,1,1] - local[0,1,1,1]
        diff[i,j,k,0,2] = local[2,1,1,2] - local[0,1,1,2]
        diff[i,j,k,1,0] = local[1,2,1,0] - local[1,0,1,0]
        diff[i,j,k,1,1] = local[1,2,1,1] - local[1,0,1,1]
        diff[i,j,k,1,2] = local[1,2,1,2] - local[1,0,1,2]
        diff[i,j,k,2,0] = local[1,1,2,0] - local[1,1,0,0]
        diff[i,j,k,2,1] = local[1,1,2,1] - local[1,1,0,1]
        diff[i,j,k,2,2] = local[1,1,2,2] - local[1,1,0,2]

    start = time.time()    
    for (i,j,k) in product(np.arange(N-2), np.arange(N-2), np.arange(N-2)):
        find_diff(n, i,j,k)
        if (j,k) == (N-3, N-3) and if_print == True:
            print(f'{i+1}/{N-2}', round(time.time()-start,2))
            start = time.time()

    diff = diff / 2 / ( width/ (N-1) )

    n = n[1:-1,1:-1,1:-1]
    splay = np.einsum( "abcii" , diff)**2
    twist_linear = np.einsum( "ijk, abck, abcij -> abc", levi, n , diff)
    twist = deform[:,:,:,3]**2
    bend_vector  = np.einsum( "ijk, lmj, abci, abclm -> abck", levi, levi, n , diff )
    bend = np.sum( bend_vector**2, axis=0 )

    deform = [splay, twist, bend, twist_linear, bend_vector]
    deform = np.array(deform).transpose((1,2,3,0))

    return deform, diff

def get_deform_Q(n, width):

    N = np.shape(n)[0]

    Q = np.einsum('abci, abcj -> abcij', n, n)
    Q = Q - np.eye(3)/3

    diffQ = np.zeros( (N, N, N, 3, 3, 3) )   # indexx, indexy, indexz, index_diff, index_Q1, indexQ2, 
    diffQ[:, :, :, 0] = np.gradient(Q, axis=0) / ( width / (N-1) )
    diffQ[:, :, :, 1] = np.gradient(Q, axis=1) / ( width / (N-1) )
    diffQ[:, :, :, 2] = np.gradient(Q, axis=2) / ( width / (N-1) )

    energy1 = np.einsum("nmlabc, nmlabc -> nml", diffQ, diffQ)
    energy2 = np.einsum("nmlaac, nmlbbc -> nml", diffQ, diffQ)
    energy3 = np.einsum("nmlab,  nmlacd, nmlbcd -> nml", Q, diffQ, diffQ)


    splay =  - energy1  +  6 * energy2  - 3 * energy3
    splay = splay / 6

    twist_linear = np.einsum("abc, nmlad, nmlbcd -> nml", levi, Q, diffQ)
    twist = twist_linear**2

    temp1 = np.einsum('nmlab, nmlbia -> nmli', Q, diffQ)
    temp2 = np.einsum('nmlia, nmlbab -> nmli', Q, diffQ)
    bend_vector = - 2 * temp1 - temp2
    bend = np.sum(bend_vector**2, -1)

    deform = [splay, twist, bend, twist_linear, bend_vector]
    deform = np.array(deform).transpose((1,2,3,0))
    deform = deform[1:-1,1:-1,1:-1]

    return deform, Q

def get_deform_Q_divide(n, width, divn = 2):

    N = np.shape(n)[0]
    divN = int(N/divn)

    terms = np.zeros((6, N, N, N))

    def enclose(x):
        if x<0:
            return 0, 0
        return x, 1
    
    def each_part(index1, index2):
        index1, pad1 = enclose(index1)
        index2, pad2 = enclose(index2)

        nnow = n[index1 : index2]
        Q = np.einsum('nmli, nmlc -> nm;ij', nnow, nnow)
        Q = Q - np.eye(3)/3

        diffQ = np.zeros( list(np.shape(Q)) +[3] )
        diffQ[:, :, :, 0] = np.gradient(Q, axis=0) / ( width / (N-1) )
        diffQ[:, :, :, 1] = np.gradient(Q, axis=1) / ( width / (N-1) )
        diffQ[:, :, :, 2] = np.gradient(Q, axis=2) / ( width / (N-1) )

        Q = Q[pad1:index2-index1-pad2]
        diffQ = diffQ[pad1:index2-index1-pad2]

        index1 += pad1
        index2 -= pad2

        terms[0, index1 : index2] = np.einsum("nmlabc, nmlabc -> nml", diffQ, diffQ)
        terms[1, index1 : index2] = np.einsum("nmlaac, nmlbbc -> nml", diffQ, diffQ)
        terms[2, index1 : index2] = np.einsum("nmlab,  nmlacd, nmlbcd -> nml", Q, diffQ, diffQ)
        terms[3, index1 : index2] = np.einsum("abc, nmlad, nmlbcd -> nml", levi, Q, diffQ)
        terms[4, index1 : index2] = np.einsum('nmlab, nmlbia -> nmli', Q, diffQ)
        terms[5, index1 : index2] = np.einsum('nmlia, nmlbab -> nmli', Q, diffQ)

    for i in range(divn-1):
        print(f'start part {i+1}')
        now = time.time()
        index1 = i*divN-1
        index2 = (i+1)*divN+1
        each_part(index1, index2)
        print(f'finished! with {round(time.time()-now, 2)}s')

    print('start the last part')
    now = time.time()
    each_part((i+1)*divN-1, N)
    print(f'finished! with {round(time.time()-now, 2)}s')

    splay =  - terms[0]  +  6 * terms[1]  - 3 * terms[2]
    splay = splay / 6

    twist_linear = terms[3]
    twist = twist_linear**2

    bend_vector = - 2 * terms[4] - terms[5]
    bend = np.sum(bend_vector**2, -1)

    deform = [splay, twist, bend, twist_linear, bend_vector]
    deform = np.array(deform).transpose((1,2,3,0))
    deform = deform[1:-1,1:-1,1:-1]

    return deform








    


