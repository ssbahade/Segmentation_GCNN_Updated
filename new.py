from pygsp import graphs, plotting
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

G2 = graphs.Grid2d(N1=28,N2=28)
G2.compute_fourier_basis()
for i_signal in range(0,5):
    G2.plot_signal(x_train[i_signal,:,:].flatten())


signal = x_train[54,:,:].flatten()
filter = np.vstack((np.ones((10,1)),np.zeros((774,1))))

signal_in_spectral = G2.gft(signal)
signal_filtered = np.multiply(signal_in_spectral, filter)

ax = plt.subplot(131);
G2.plot_signal(signal, ax=ax);
#G2.set_coordinates('line1D');
plt.subplot(132)
plt.plot(signal_in_spectral);
plt.subplot(133)
plt.plot(signal_filtered);
#fig.tight_layout();
plt.show()