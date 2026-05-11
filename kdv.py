# Script courtesy of Dr. Alex Bihlo
# contact Alex (abihlo@mun.ca) if you want to use this script for purpose 
# other than learning in the contex of DAAO course

# Minor modifications by Sergiy Vasylkevych


import tensorflow as tf
import numpy as np
from pyDOE import lhs
import matplotlib.pyplot as plt


tf.keras.backend.set_floatx('float64')
nu = -0.0025
# Boundaries of the computational domain
xleft, xright = 7, 10
t0, tfinal = 0, 1.0
u0 = lambda x: np.cos(2*np.pi*(x-xleft)/(xright-xleft))

# Plot initial condition
x = np.linspace(xleft, xright, 100)
figin=plt.figure()
plt.plot(x, u0(x))
plt.grid()
plt.xlabel('x')
plt.ylabel('u')
plt.pause(0.1)   # makes sure that the plot is shown before execution proceeds

class Periodic(tf.keras.layers.Layer):
  def __init__(self):
    super(Periodic, self).__init__()

  def call(self, inputs):
    return tf.concat((tf.cos(np.pi*inputs), tf.sin(np.pi*inputs)), axis=1)

# Define the normalization layer
class Normalize(tf.keras.layers.Layer):
  def __init__(self, xmin, xmax):
    super(Normalize, self).__init__()
    self.xmin = xmin
    self.xmax = xmax

  def call(self, inputs):
    return 2.0*(inputs-self.xmin)/(self.xmax-self.xmin)-1.0

  def get_config(self):   
  # this is needed in order to save/load the model because
  # __init__ for this layer has  parameters. 
        config = super().get_config().copy()
        config.update({
            'xmin': self.xmin,
            'xmax': self.xmax })
        return config

# Define the network
n_layers = 6
n_units = 40

inp1 = tf.keras.layers.Input(shape=(1,))
b1 = Normalize(t0, tfinal)(inp1)

inp2 = tf.keras.layers.Input(shape=(1,))
b21= Normalize(xleft, xright)(inp2)
b2 = Periodic()(b21)
b = tf.keras.layers.Concatenate()([b1, b2])

for i in range(n_layers):
  b = tf.keras.layers.Dense(n_units, activation='tanh', kernel_initializer='glorot_normal')(b)
out = tf.keras.layers.Dense(1)(b)

model = tf.keras.models.Model([inp1, inp2], out)
model.summary()

@tf.function
def trainStep(t_pde, x_pde, t_init, x_init, u_init, nu):

  # Outer gradient for tuning network parameters
    with tf.GradientTape() as tape:

      # # Inner gradient for derivatives of u wrt x and t
      with tf.GradientTape(persistent=True) as tape1:
        tape1.watch(t_pde), tape1.watch(x_pde)
        u = model([t_pde, x_pde])
        [ut, ux] = tape1.gradient(u, [t_pde, x_pde])  
        uxx = tape1.gradient(ux, x_pde)
      uxxx =tape1.gradient(uxx, x_pde)
      del tape1
      # Solve the KdV equations
      eqn = ut+u*ux-nu*uxxx

      # Define the PDE loss  
      PDEloss = tf.reduce_mean(tf.square(eqn))

      # Define the initial value loss
      u_init_pred = model([t_init, x_init])      
      IVloss = tf.reduce_mean(tf.square(u_init-u_init_pred))

      # Global loss
      loss = PDEloss + IVloss

    # Compute the gradient of the global loss wrt the model parameters
    grads = tape.gradient(loss, model.trainable_variables)

    return PDEloss, IVloss, grads

def PNNtrain(t_bdry, x_bdry, u0, nu, epochs=5000, N_pde=10000, N_iv=100):

  # Optimizer to be used
  lr = tf.keras.optimizers.schedules.PolynomialDecay(1e-3, epochs, 1e-4)
  opt = tf.keras.optimizers.Adam(lr)

  # Sample points where to evaluate the PDE
  tx_min = np.array([t_bdry[0], x_bdry[0]])
  tx_max = np.array([t_bdry[1], x_bdry[1]])  
  pde_points = tx_min + (tx_max - tx_min)*lhs(2, N_pde)
  t_pde = pde_points[:,0]
  x_pde = pde_points[:,1]


  # Sample points where to evaluate the initial values
  init_points = tx_min[1:] + (tx_max[1:] - tx_min[1:])*lhs(1, N_iv)
  x_init = init_points
  t_init = t_bdry[0]+ 0.0*x_init
  u_init = u0(x_init)
  inits = np.column_stack([t_init, x_init, u_init])

  # Create Tensorflow datasets from data points

  ds_pde = tf.data.Dataset.from_tensor_slices(pde_points)
  ds_pde = ds_pde.cache().shuffle(N_pde).batch(N_pde)

  ds_init = tf.data.Dataset.from_tensor_slices(inits)
  ds_init = ds_init.cache().shuffle(N_iv).batch(N_iv)

  ds = tf.data.Dataset.zip((ds_pde, ds_init))
  ds = ds.prefetch(tf.data.AUTOTUNE)
 
  # Epoch loss initialization
  epoch_loss = np.zeros(epochs)

  # Main training loop
  for i in range(epochs):

    for (pdes, inits) in ds:
      
      t_pde, x_pde = pdes[:,:1], pdes[:,1:2]
      t_init, x_init, u_init = inits[:,:1], inits[:,1:2], inits[:,2:3]
      PDEloss, IVloss, grads = trainStep(t_pde, x_pde, t_init, x_init, u_init, nu)
    
      # Gradient step
      opt.apply_gradients(zip(grads, model.trainable_variables))

      epoch_loss[i] += PDEloss + IVloss

    if (np.mod(i, 100)==0):
      print("PDE loss, IV loss in {}th epoch: {: 6.4f}, {: 6.4f}.".format(i, PDEloss.numpy(), IVloss.numpy()))
    
  return epoch_loss

epochs = 10000
loss = PNNtrain([t0, tfinal], [xleft, xright], u0, nu, epochs)

figtr=plt.figure()
plt.semilogy(range(0, epochs), loss)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
plt.savefig('loss.png')
plt.pause(0.1)


# Grid where to evaluate the model
l, m = 100, 200
t = np.linspace(t0, tfinal, l)
x = np.linspace(xleft, xright, m)
T, X = np.meshgrid(t, x, indexing='ij')

u = model([T.flatten()[:,np.newaxis], X.flatten()[:,np.newaxis]])
u = np.reshape(u, (l, m))
 
#------------------------------Visualization---------- 
import matplotlib.animation as animation
nt=l
anifname="kdv.avi"
sf=1
spf=''
fig=plt.figure(figsize=(10,5))

def plot_kdv(i):
  fig.clf()
  ax=fig.add_subplot(111)
  ax.plot(X[i,], u[i,])
  ax.set_title('Neural network solution at time {: 4.2f}'.format(T[i,0]))
  ax.set_ylabel('u(t)')
  ax.set_xlabel('x')
  plt.pause(0.1)
  
def init():
    return plot_kdv(0)

def animate(frame):
    return plot_kdv(frame) 

def plot_Hov():
    fig1=plt.figure(figsize=(6,5))
    plt.contourf(X, T, u, 100, cmap=plt.cm.jet)
    plt.title('Neural network solution')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.colorbar()
  
print('<======================>')
plot_Hov()
bcont=True
while (bcont) :
    print('Input number between 0 and %i to display corresponding timestep;' %(nt-1))
    print('-4 to display animation (can not be saved) ')
    print('-3 to save snapshots')
    print('-2 to generate and save animation')
    tsn=input('-1 to exit: ')
    tsn=int(tsn)
    if (tsn==-1):
        bcont=False
        print('Type "plot_kdv(n)" to display the forecast at saved time step n or "exit" to exit the ipython console')
    elif (tsn==-2):
        print('Generating animation')
        fig.clf()
        ani = animation.FuncAnimation(fig, animate, nt, interval=2, blit=False,
                          init_func=init, repeat=False)
        ani.save(anifname, writer=animation.FFMpegWriter(fps=16)) 
    elif (tsn==-3):
        for i in range(0,nt,sf):
            plot_kdv(i) 
            plt.savefig(spf+'kdv'+str(i)+'.png')
    elif (tsn==-4):
         for i in range(0,nt,5):
             plot_kdv(i) 
    else:
        plot_kdv(tsn)
        plt.pause(0.01)



