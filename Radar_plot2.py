# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 10:16:41 2016

@author: Tavo
"""
import os
import numpy as np
import pandas as pd
from pandas import Series, DataFrame, Panel
from datetime import datetime
import random as random

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show, rc

from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection

import seaborn as sns

bike = pd.read_csv('bike_project/train.csv')
bike = bike.dropna()
bike.datetime = pd.to_datetime(bike.datetime)
bike.set_index('datetime', inplace=True)
bike['weekday'] = bike.index.weekday.astype('int')

bike_dsum = bike.resample('D').sum()
bike_dmean = bike.resample('D').mean()

#bike_dmean['gamma'] = ((2*np.pi/7)*bike_dmean.weekday) + np.pi/2 

bike_dmean['gamma'] = ((2*np.pi/7)*bike_dmean.weekday) + np.pi/2

bike_dmean['ran'] = np.random.choice(range(-15,15), bike_dmean.shape[0])
bike_dmean.ran = bike_dmean.ran / 100

gamma = bike_dmean.gamma + bike_dmean.ran
r = bike_dsum['count']
t = bike_dsum['casual']

'''=============================='''



def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    # rotate theta such that the first axis is at the top
    theta += np.pi/2

    def draw_poly_patch(self):
        verts = unit_poly_verts(theta)
        return plt.Polygon(verts, closed=True, edgecolor='k')

    def draw_circle_patch(self):
        # unit circle centered on (0.5, 0.5)
        return plt.Circle((0.5, 0.5), 0.5)

    patch_dict = {'polygon': draw_poly_patch, 'circle': draw_circle_patch}
    if frame not in patch_dict:
        raise ValueError('unknown value for `frame`: %s' % frame)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1
        # define draw_frame method
        draw_patch = patch_dict[frame]

        def fill(self, *args, **kwargs):
            """Override fill so that line is closed by default"""
            closed = kwargs.pop('closed', True)
            return super(RadarAxes, self).fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super(RadarAxes, self).plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            return self.draw_patch()

        def _gen_axes_spines(self):
            if frame == 'circle':
                return PolarAxes._gen_axes_spines(self)
            # The following is a hack to get the spines (i.e. the axes frame)
            # to draw correctly for a polygon frame.

            # spine_type must be 'left', 'right', 'top', 'bottom', or `circle`.
            spine_type = 'circle'
            verts = unit_poly_verts(theta)
            # close off polygon by repeating first vertex
            verts.append(verts[0])
            path = Path(verts)

            spine = Spine(self, spine_type, path)
            spine.set_transform(self.transAxes)
            return {'polar': spine}

    register_projection(RadarAxes)
    return theta


def unit_poly_verts(theta):
    """Return vertices of polygon for subplot axes.

    This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
    """
    x0, y0, r = [0.5] * 3
    verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
    return verts

'''====plots===='''
n = 0
imges = []
spoke_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

start_date = pd.datetime(2011, 1, 1)
end_date = pd.datetime(2011, 12, 31)

for single_date in pd.date_range(start_date, end_date, freq='M'):
    dd = single_date.strftime("%Y-%m-%d")
    gammax = gamma['2011-1-1':dd]
    rx = r['2011-1-1':dd]
    tx = t['2011-1-1':dd]
    n += 1 
  
    
    if __name__ == '__main__':
        N = 7
        theta = radar_factory(N, frame='circle')
        fig = plt.figure(figsize=(6, 6))
        fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)
        ax = fig.add_subplot(111, projection='radar')
        plt.rgrids([2000, 4000, 6000, 10000])
        ax.set_title('Year 2011', weight='bold', size='medium', position=(0.5, 1.1),\
        horizontalalignment='center', verticalalignment='center')
        
        colors = ['b', 'r', 'g', 'm', 'y']
        
        ax.scatter(gammax, rx, c=tx, s=100)
        ax.fill(gammax, rx, facecolor='b', alpha=0.1)
        ax.set_varlabels(spoke_labels)
        ax.set_rmax(6500.0)
    
        plt.subplot(1, 1, 1)
        labels = ('Factor 1', 'Factor 2')
        legend = plt.legend(labels, loc=(0.9, .95), labelspacing=0.1)
        plt.setp(legend.get_texts(), fontsize='small')

        plt.figtext(0.5, 0.965, 'Number of bikes hired',
           ha='center', color='black', weight='bold', size='large')
        plt.show()
        #imges = imges.append(('graph_{0}.png'.format(str(n))))
        fig.savefig('graph_{0}'.format(str(n)))
                
'=====Create GIF from the saved images====='
#!/usr/bin/env python
'''
from natsort import natsorted, ns
#natsorted(x, alg=ns.IGNORECASE)  # or alg=ns.IC
file_names = sorted((fn for fn in os.listdir('.') if fn.endswith('.png')))
file_names = natsorted(file_names, alg=ns.IGNORECASE)

import imageio
im = imageio.mimread(file_names)
im.shape
imageio.imwrite('chelsea-gray.gif', im[:, :, 0])'''


#file_names = sorted(['graph_1.png', 'graph_2.png', 'graph_3.png', 'graph_4.png'])

