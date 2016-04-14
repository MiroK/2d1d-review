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


def table(files):
    if isinstance(files, str):
        files = [files]

    all_data = []
    for f in files:
        data = np.loadtxt(f)
        all_data.append(data[:, -1]-1)  # nvertices -> ncells
        all_data.append(data[:,  2])
        all_data.append(data[:,  3])

    if len(files) > 1:
        delete = []
        for i in range(len(files)-1):
            assert np.linalg.norm(all_data[i*3] - all_data[(i+1)*3]) < 1E-13
            delete.append((i+1)*3)
        for row in reversed(delete): del all_data[row]


    # row formats
    n_fmt = lambda v: str(int(np.log2(v)))
    t_fmt = lambda v: '%.3f' % v
    r_fmt = lambda v: '%.3f' % v
    
    all_data = np.array(all_data)
    for i, row in enumerate(all_data):
        if i == 0:
            fmt_row = map(n_fmt, row[1:])
        elif i in (1, 3):
            fmt_row = map(t_fmt, row[1:])
        else:
            fmt_row = map(r_fmt, row[1:])

        print ' & '.join(fmt_row) + r'\\'



# ----------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    if sys.argv[1] == 'plot': plot(sys.argv[2:])

    if sys.argv[1] == 'table': table(sys.argv[2:])
