import matplotlib
matplotlib.use('Agg')
from matplotlib import cm, colors
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

def alt1_pdf(x):
    return 0.48 * norm.pdf(x, -2, np.sqrt(1)) + 0.04 * norm.pdf(x, 0, np.sqrt(16)) + 0.48 * norm.pdf(x, 2, np.sqrt(1))

def alt2_pdf(x):
    return 0.4 * norm.pdf(x, -1.25, np.sqrt(2)) + 0.2 * norm.pdf(x, 0, np.sqrt(4)) + 0.4 * norm.pdf(x, 1.25, np.sqrt(2))

def alt3_pdf(x):
    return 0.3 * norm.pdf(x, 0, np.sqrt(0.1)) + 0.4 * norm.pdf(x, 0, np.sqrt(1)) + 0.3 * norm.pdf(x, 0, np.sqrt(9))

def alt4_pdf(x):
    return 0.2 * norm.pdf(x, -3, np.sqrt(0.01)) + 0.3 * norm.pdf(x, -1.5, np.sqrt(0.01)) + 0.3 * norm.pdf(x, 1.5, np.sqrt(0.01)) + 0.2 * norm.pdf(x, 3, np.sqrt(0.01))

def alt1_noisy_pdf(x):
    return 0.48 * norm.pdf(x, -2, np.sqrt(1+1)) + 0.04 * norm.pdf(x, 0, np.sqrt(16+1)) + 0.48 * norm.pdf(x, 2, np.sqrt(1+1))

def alt2_noisy_pdf(x):
    return 0.4 * norm.pdf(x, -1.25, np.sqrt(2+1)) + 0.2 * norm.pdf(x, 0, np.sqrt(4+1)) + 0.4 * norm.pdf(x, 1.25, np.sqrt(2+1))

def alt3_noisy_pdf(x):
    return 0.3 * norm.pdf(x, 0, np.sqrt(0.1+1)) + 0.4 * norm.pdf(x, 0, np.sqrt(1+1)) + 0.3 * norm.pdf(x, 0, np.sqrt(9+1))

def alt4_noisy_pdf(x):
    return 0.2 * norm.pdf(x, -3, np.sqrt(0.01+1)) + 0.3 * norm.pdf(x, -1.5, np.sqrt(0.01+1)) + 0.3 * norm.pdf(x, 1.5, np.sqrt(0.01+1)) + 0.2 * norm.pdf(x, 3, np.sqrt(0.01+1))


def alt1_sample():
    u = np.random.random()
    if u <= 0.48:
        return np.random.normal(-2, np.sqrt(1))
    elif u <= 0.52:
        return np.random.normal(0, np.sqrt(16))
    return np.random.normal(2, np.sqrt(1))

def alt2_sample():
    u = np.random.random()
    if u <= 0.4:
        return np.random.normal(-1.25, np.sqrt(2))
    elif u <= 0.6:
        return np.random.normal(0, np.sqrt(4))
    return np.random.normal(1.25, np.sqrt(2))

def alt3_sample():
    u = np.random.random()
    if u <= 0.3:
        return np.random.normal(0, np.sqrt(0.1))
    elif u <= 0.7:
        return np.random.normal(0, np.sqrt(1))
    return np.random.normal(0, np.sqrt(9))
    
def alt4_sample():
    u = np.random.random()
    if u <= 0.2:
        return np.random.normal(-3, np.sqrt(0.01))
    elif u <= 0.5:
        return np.random.normal(-1.5, np.sqrt(0.01))
    elif u <= 0.8:
        return np.random.normal(1.5, np.sqrt(0.01))
    return np.random.normal(3, np.sqrt(0.01))
    
def test_pdf(x):
    return norm.pdf(x, 3, 1)

def test_sample():
    return np.random.normal(3, 1)

def plot_alts(filename):
    fig, axarr = plt.subplots(2,2)

    x = np.linspace(-6, 6, 1000)

    # Alt 1
    axarr[0,0].plot(x, pdf_alt1(x))
    axarr[0,0].set_title('Case 1')

    # Alt 2
    axarr[1,0].plot(x, pdf_alt2(x))
    axarr[1,0].set_title('Case 2')

    # Alt 3
    axarr[0,1].plot(x, pdf_alt3(x))
    axarr[0,1].set_title('Case 3')

    # Alt 4
    axarr[1,1].plot(x, pdf_alt4(x))
    axarr[1,1].set_title('Case 4')

    plt.savefig(filename, bbox_inches='tight')

if __name__ == '__main__':
    plot_alts('synthetic_alt_densities.pdf')