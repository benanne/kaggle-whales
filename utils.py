import numpy as np
import matplotlib.pyplot as plt
plt.ion()
    
    
class progress_plot(object):
    """
    Usage:
    p = progress_plot(color='r')
    p(5)
    p(4)
    p(2)
    """
    def __init__(self, num, *args, **kwargs):
        self.num = num
        self.args = args
        self.kwargs = kwargs
        self.values = []
        self._figure = None
        
    def __call__(self, *values):
        self.values.append(values)
        self.plot()
    
    @property
    def figure(self):
        if self._figure is None:
            self._figure = plt.figure(self.num)
        return self._figure
        
    @property
    def series(self):
        return zip(*self.values)
        
    def plot(self):
        plt.figure(self.num)
        plt.clf()
        for s in self.series:
            plt.plot(s, *self.args, **self.kwargs)
        plt.draw()
        plt.show()
        


def most_square_shape(num_blocks, blockshape=(1,1)):
    x, y = blockshape
    num_x = np.ceil(np.sqrt(num_blocks * y / float(x)))
    num_y = np.ceil(num_blocks / num_x)
    return (num_x, num_y)
 
 
 
def visualise_filters(data, height=8, width=8, posneg=False):
    """
    input: a (height*width) x H matrix, which is reshaped into filters
    """
    num_x, num_y = most_square_shape(data.shape[1], (height, width))
    
    #pad with zeros so that the number of filters equals num_x * num_y
    padding = np.ones((height*width, num_x*num_y - data.shape[1])) * data.min()
    data_padded = np.hstack([data, padding])
    
    data_split = data_padded.reshape(height, width, num_x, num_y)
    
    data_with_border = np.ones((height+1, width+1, num_x, num_y)) * data.min()
    data_with_border[:height, :width, :, :] = data_split
    
    filters = data_with_border.transpose(2,0,3,1).reshape(num_x*(height+1), num_y*(width+1))
    
    filters_with_left_border = np.zeros((num_x*(height+1)+1, num_y*(width+1)+1))
    filters_with_left_border[1:, 1:] = filters
    
    if posneg:
        m = np.abs(data).max()
        plt.imshow(filters_with_left_border, interpolation='nearest', cmap=plt.cm.RdBu, vmin=-m, vmax=m)
    else:
        plt.imshow(filters_with_left_border, interpolation='nearest', cmap=plt.cm.binary, vmin = data.min(), vmax=data.max())
        
        
def visualise_filters_scaled(data, dim=28):
    """
    input: a (dim*dim) x H matrix, which is reshaped into filters. Each filter is rescaled to have values in [0,1]
    """
    num_x, num_y = most_square_shape(data.shape[1], (dim, dim))
    
    # rescale data
    data = data - data.min(0).reshape(1, -1)
    data = data / data.max(0).reshape(1, -1)
    
    #pad with zeros so that the number of filters equals num_x * num_y
    padding = np.zeros((dim*dim, num_x*num_y - data.shape[1]))    
    data_padded = np.hstack([data, padding])
    
    data_split = data_padded.reshape(dim, dim, num_x, num_y)
    
    data_with_border = np.zeros((dim+1, dim+1, num_x, num_y))
    data_with_border[:dim, :dim, :, :] = data_split
    
    filters = data_with_border.transpose(2,0,3,1).reshape(num_x*(dim+1), num_y*(dim+1))
    
    filters_with_left_border = np.zeros((num_x*(dim+1)+1, num_y*(dim+1)+1))
    filters_with_left_border[1:, 1:] = filters
    
    plt.imshow(filters_with_left_border, interpolation='nearest', cmap=plt.cm.gray, vmin=0, vmax=1)