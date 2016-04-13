import matplotlib.pyplot as plt
import numpy as np

def plot(files):
    if isinstance(files, str):
        files = [files]

    fig = plt.figure()
    ax = fig.gca()
    eigs = []
    for f in files:
        data = np.loadtxt(f)
        
        sizes = data[:, 0]
        times = data[:, 1]
        ax.loglog(sizes, times, label=f)

        print f, 'last 3 rates', data[-3:, 2]
        
        eigs.append(data[:, -1])

    if len(eigs) > 1:
        for i in range(len(eigs)):
            for j in range(i):
                print np.linalg.norm(eigs[i]-eigs[j]), '@', i, j

    ax.set_xlabel("n")
    ax.set_ylabel("[s]")
    # Rates
    ax.legend(loc='best')

    plt.show()

# ----------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    plot(sys.argv[1:])
