from pygsp import graphs,filters, plotting
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
img = Image.open('C:/Users/sachin/PycharmProjects/SegmentationNew_Method/runs/1548301457.980771/test100.png')
img = np.asarray(img)
img = np.resize(img, (8,8,3))
#plt.imshow(img)
#plt.show()

G = graphs.NNGraph(1024,3)
G.compute_differential_operator()
G.compute_fourier_basis()
G.plotting['vertex_size'] = 20
G.plot()

#img = np.reshape(img,(-1,3))
#G.set_coordinates(kind='spring',seed=42)
for i_signal in range(0, len(img)):
    G.plot_signal(img[i_signal])


fig, axes = plt.subplots(1, 2)
_ = axes[0].spy(G.W, markersize=2)
G.plot(ax=axes[0])



print("finfish")