#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 17:14:40 2021

@author: rjedicke
"""

import matplotlib as mpl
import matplotlib.pyplot as pyplot
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import Normalize

import numpy as np

from scipy.ndimage.interpolation import shift

    
    


#-----------------------------------------------------------------------------------------------
# used to be plot2d_multi

def plot2d(
    x, y, 
    figure=None, axis=None,
    title='', size=(8, 8), dpi=200,
    xError=None, yError=None, errorLineWidth=1, errorBarColor=None,
    xlabel='', xrange=(), logx=False,
    ylabel='', yrange=(), logy=False,
    markerColor=None, markersize=1, markertype='.',
    linestyle='none', linewidth=2, lineColor=None, alpha=1.0,
    axislabelfontsize=12, ticklabelfontsize=10, legendfontsize=10,
    dataLabel=None, bLegend=False, legendFractionalLocation=None,
    bShow=True, saveto=''
):
    """
    Plots 2D data with optional error bars, log scaling, and labeling.

    Parameters:
        x, y : array-like
            Data for the X and Y axes.
        figure, axis : matplotlib Figure and Axes (optional)
            If not provided, new figure and axis will be created.
        title : str
            Title of the plot.
        size : tuple
            Figure size in inches (width, height).
        dpi : int
            Dots per inch (plot resolution).
        xError, yError : array-like or None
            Error values for X and Y axes. (neg,pos)
        errorLineWidth : float
            Width of error bar lines.
        errorBarColor : str or None
            Color of error bars.
        xlabel, ylabel : str
            Axis labels.
        xrange, yrange : tuple
            Axis limits for X and Y axes.
        logx, logy : bool
            If True, use logarithmic scale on respective axis.
        markerColor : str
            Color of marker face and edge.
        markersize : float
            Size of the marker.
        markertype : str
            Marker style, e.g., '.', 'o', '^'.
        linestyle : str
            Line style connecting data points.
        linewidth : float
            Width of the connecting line.
        lineColor : str
            Color of the line.
        alpha : float
            Transparency of the plot elements (0 to 1).
        axislabelfontsize : int
            Font size for axis labels.
        ticklabelfontsize : int
            Font size for tick labels.
        legendfontsize : int
            Font size for legend.
        dataLabel : str
            Label for the dataset, shown in legend.
        bShow : bool
            Whether to display the plot immediately.
        bLegend : bool
            Whether to include a legend.
        saveto : str
            If provided, save the figure to this file path.

    Returns:
        figure, axis : Matplotlib Figure and Axes objects
    """

    # Create a new figure and axis if none provided
    if figure is None and axis is None:
        figure, axis = pyplot.subplots(figsize=size, dpi=dpi)

    # Main plot with optional error bars and markers
    axis.errorbar(
        x, y,
        xerr=xError, yerr=yError,
        elinewidth=errorLineWidth, ecolor=errorBarColor,
        linestyle=linestyle, lw=linewidth, color=lineColor, alpha=alpha,
        marker=markertype, markersize=markersize,
        markerfacecolor=markerColor, markeredgecolor=markerColor,
        label=dataLabel
    )

    # Axis labels
    axis.set_xlabel(xlabel, fontsize=axislabelfontsize)
    axis.set_ylabel(ylabel, fontsize=axislabelfontsize)

    # Tick label size
    axis.tick_params(axis='both', which='major', labelsize=ticklabelfontsize)

    # Title
    axis.set_title(title)

    # Log scales if requested
    if logx:
        axis.set_xscale("log")
    if logy:
        axis.set_yscale("log")

    # Axis limits
    if xrange:
        axis.set_xlim(xrange)
    if yrange:
        axis.set_ylim(yrange)

    # Legend
    if bLegend:
        axis.legend( fontsize=legendfontsize, loc=legendFractionalLocation )

    # Save to file if a path is provided
    if saveto:
        figure.savefig(saveto, bbox_inches='tight')

    # Show the plot if requested
    if bShow:
        pyplot.show()

    return figure, axis

    
    


#-----------------------------------------------------------------------------------------------
# used to be plot2d_multi
def plot2d_onExisting( x, y, figure=None, axis=None, xError=None, yError=None, title='', size=(8,8), dpi=200, 
                 xlabel='', xrange=(), logx=False, 
                 ylabel='', yrange=(), logy=False, 
                 axislabelfontsize=12, linestyle='none', color=None, markersize=1, markertype='.', dataLabel=None, 
                 bShow=True, bLegend=False, saveto='' ):
   
    axis.errorbar( x, y, xerr=xError, yerr=yError, color=color, linestyle=linestyle, marker=markertype, markersize=markersize, label=dataLabel )
    
    axis.set_xlabel( xlabel, fontsize=axislabelfontsize )    
    axis.set_ylabel( ylabel, fontsize=axislabelfontsize )
    axis.set_title(  title )
        
    if( logy   ):   axis.set_yscale( "log" )
    if( logx   ):   axis.set_xscale( "log" )
    
    if( xrange ):   axis.set_xlim( xrange )
    if( yrange ):   axis.set_ylim( yrange )
    
    if( bLegend ):  axis.legend()
    
    if( saveto ):   figure.savefig( saveto, bbox_inches='tight' )
    
    if( bShow  ):   figure.show()

    return figure, axis
    
    


#-----------------------------------------------------------------------------------------------
# used to be plot2d
def plot2d_20231101( x, y, xError=None, yError=None, xlabel='', xrange=(), ylabel='', title='', yrange=(), linestyle='none', 
           markersize=1, markertype='.', size=(8,8), dpi=200, bShow=True, xfunc=(), yfunc=(),
           logx=False, logy=False, saveto='' ):

    if( bShow ):
    
        pyplot.figure( figsize=size, dpi=dpi )
        
        pyplot.errorbar( x, y, xerr=xError, yerr=yError, linestyle=linestyle, marker=markertype, markersize=markersize )
        
        pyplot.xlabel( xlabel )    
        pyplot.ylabel( ylabel )
        pyplot.title(  title )
        
        if( logy   ):  pyplot.yscale( "log" )
        if( logx   ):  pyplot.xscale( "log" )
        
        if( xrange ):  pyplot.xlim( xrange )
        if( yrange ):  pyplot.ylim( yrange )
        
        if( len(xfunc) != 0 ): pyplot.plot( xfunc, yfunc, 'g-' )  # plot an additional function if defined
        
        if( saveto ):  pyplot.savefig( saveto, bbox_inches='tight' )
        
        pyplot.show()
        pyplot.close()
       
    
    
    
    
#-----------------------------------------------------------------------------------
# plots  npn vs npx as if the data is a histogram instead of histogramming the array
# assumes that npx is the bin edges returned by np.hist
def hist_multi( npBinEdges, npValues, figure=None, title='', xrange=(), xlabel="", xticks=(), yrange=(), ylabel='value', 
                   label='blah', logx=False, logy=False, bShow=True,
                    bShowErrorBars=False, bPoissonErrors=False, bUseYErrorVectors=False, yErrorNeg=[], yErrorPos=[],
                    plotFunc=True, xfunc=(), yfunc=(), figsize=(8,8), dpi=200, saveto='' ):
    
    if( not figure ): figure, ax = pyplot.subplots( figsize=figsize, dpi=dpi )
    
    if( xrange ): pyplot.xlim( xrange )
    if( yrange ): pyplot.ylim( yrange )
        
    npValuesNoNaN = np.nan_to_num( npValues )
    
    pyplot.hist( npBinEdges[:-1], npBinEdges, weights=npValuesNoNaN, histtype="step", label=label )
    
    if( logy ): pyplot.yscale( "log" )
        
    if( logx ): pyplot.xscale( "log" )
    
    binCenters = (shift(npBinEdges,-1)[:-1]+npBinEdges[:-1])/2
        
    if( bShowErrorBars ):
        if( bPoissonErrors ):  yError = np.sqrt( npValues );  yErrorPos = yError;  yErrorNeg = yError         
        pyplot.errorbar( binCenters, npValues, yerr=[ yErrorNeg, yErrorPos ], fmt='none' )

    pyplot.xlabel( xlabel )
    pyplot.ylabel( ylabel )

    if( plotFunc ):  pyplot.plot( xfunc, yfunc, 'g-' )
    
    if( xticks ):  pyplot.xticks( xticks[0], xticks[1] )

    if( title != '' ):  pyplot.title( title )
    
    pyplot.legend()
    
    if( saveto ):   pyplot.savefig( saveto, bbox_inches='tight' )
    
    if( bShow ):
        pyplot.show()
        pyplot.close()

    return figure
     




#-----------------------------------------------------------------------------------------------
# used to be plot3d
def plot3d( x, y, z, datalabel='data', xlabel='', xrange=(), ylabel='', yrange=(), zlabel='', zrange=() ):
      
    mpl.rcParams['legend.fontsize'] = 10
    
    fig = pyplot.figure( figsize=(5,5), dpi=200 )
    
    ax = fig.add_subplot( 111, projection='3d' )
    
    ax.plot( x, y, z, label=datalabel )

    ax.legend()
    
    ax.set_xlabel( xlabel )
    ax.set_ylabel( ylabel )
    ax.set_zlabel( zlabel )

    ax.set_xlim( xrange )    
    ax.set_ylim( yrange )
    ax.set_zlim( zrange )
    
    pyplot.show()
    
    
    
    
#-----------------------------------------------------------------------------------------------
# plot an arrow from (x0,y0) to (x1,y1)

def arrow( fig, axis, x0, y0, x1, y1, arrowstyle='->', mutation_scale=20, color='blue' ):
    
    # dx = x1 - x0
    # dy = y1 - y0

   #axis.arrow( x0, y0, dx, dy, linewidth=0.0001, width=tail_width, head_width=head_width, head_length=head_length, overhang=overhang, fc=fc, ec=ec ) 

    # axis.add_patch( pyplot.FancyArrow( x=x0, y=y0, dx=dx, dy=dy, width=tail_width5, color='blue') )
    
    axis.add_patch( FancyArrowPatch( (x0,y0), (x1,y1), arrowstyle=arrowstyle, mutation_scale=mutation_scale, color=color ) )




#-----------------------------------------------------------------------------------------------
def plot2horiz( x1, y1, x2, y2, x1label='', x1range=(), y1label='', y1range=(),
                                x2label='', x2range=(), y2label='', y2range=()):
    
    plot2d( x1, y1, x1label, x1range, y1label, y1range, size=(8,4), dpi=200, bShow=False )
    plot2d( x2, y2, x2label, x2range, y2label, y2range, size=(8,4), dpi=200, bShow=True  )





#-----------------------------------------------------------------------------------------------
def vertical_line( fig, axis, x, yRange, color, dataLabel='', linestyle='-', linewidth=0.5 ):

    xRange = ( x, x )
    
    fig, axis = plot2d( xRange, yRange, figure=fig, axis=axis, lineColor=color, linestyle=linestyle, linewidth=linewidth, dataLabel=dataLabel, bShow=False )

    return fig, axis





#-----------------------------------------------------------------------------------------------
def horizontal_line( fig, axis, y, xRange, color, dataLabel='', linestyle='-', linewidth=0.5 ):

    yRange = ( y, y )
    
    fig, axis = plot2d( xRange, yRange, figure=fig, axis=axis, lineColor=color, linestyle=linestyle, linewidth=linewidth, dataLabel=dataLabel, bShow=False )

    return fig, axis





#-----------------------------------------------------------------------------------------------
def aitoff( lon_deg, lat_deg, figure=None, axis=None, title='', size=(8,5), dpi=200, 
           xError=None, yError=None, errorLineWidth=1, errorBarColor=None,
                 xlabel='', xrange=(), logx=False, 
                 ylabel='', yrange=(), logy=False, 
                 markerColor='b', markersize=1, markertype='.',
                 linestyle='none', linewidth=2, lineColor=None,
                 axislabelfontsize=8, dataLabel=None, 
                 bShow=True, bLegend=False, saveto='' ):
    
    fig = pyplot.figure( figsize=size, dpi=200 )
    
    axis = fig.add_subplot( 111, projection='aitoff' )
    
    longitudes = np.deg2rad( lon_deg ) 
    latitudes  = np.deg2rad( lat_deg )
    
    # Plot the data on the Aitoff projection
    axis.plot( longitudes, latitudes, markertype, color=markerColor, markersize=markersize )
    
    axis.tick_params( axis='x', labelsize=axislabelfontsize )  # Change the x-axis tick label size
    axis.tick_params( axis='y', labelsize=axislabelfontsize )  # Change the y-axis tick label size
    
    # Customize the grid and labels
    axis.grid( True )
    axis.set_title( title, fontsize=14 )
    
    # Show the plot
    pyplot.show()





#-----------------------------------------------------------------------------------------------
def aitoff_colorbar( lon_deg, lat_deg, values, colormap='viridis', colorbarLabel='', figure=None, axis=None, title='', titlefontsize=10, size=(8,5), dpi=200, 
           xError=None, yError=None, errorLineWidth=1, errorBarColor=None,
                 xlabel='', xrange=(), logx=False, 
                 ylabel='', yrange=(), logy=False, 
                 vminmax=(), bWhiteOnBlack=False,
                 markerColor='b', markersize=1, markertype='.',
                 linestyle='none', linewidth=2, lineColor=None,
                 axislabelfontsize=8, dataLabel=None, 
                 bShow=True, bLegend=False, saveto='' ):
    
    fig = pyplot.figure( figsize=size, dpi=200 )
    
    axis = fig.add_subplot( 111, projection='aitoff' )
    
    longitudes = np.deg2rad( lon_deg ) 
    latitudes  = np.deg2rad( lat_deg )
    
    if( vminmax ):
        norm=Normalize(vmin=vminmax[0], vmax=vminmax[1])
    else:
        norm = None 
    
    # Plot the data on the Aitoff projection
  #axis.plot( longitudes, latitudes, markertype, color=markerColor, markersize=markersize )
   #sc = axis.scatter( longitudes, latitudes, c=values, vmin=min(values), vmax=max(values), cmap=colormap, s=markersize )
    sc = axis.scatter( longitudes, latitudes, c=values, cmap=colormap, s=markersize, norm=norm )

    # Add a colorbar to indicate what the color represents (e.g., temperature)
    cbar = pyplot.colorbar( sc, ax=axis, orientation='horizontal', pad=0.1 )
    cbar.set_label( colorbarLabel ) 
    
    axis.tick_params( axis='x', labelsize=axislabelfontsize )  # Change the x-axis tick label size
    axis.tick_params( axis='y', labelsize=axislabelfontsize )  # Change the y-axis tick label size
    
    # Customize the grid and labels
    axis.grid( True )
    axis.set_title( title, fontsize=titlefontsize )

    if( bWhiteOnBlack ):
        
        axis.set_title( title, fontsize=titlefontsize, color='white' )

        fig.patch.set_facecolor('black')  # Set the figure background to black
        axis.set_facecolor('black')  # Set the plot background to black
       
        # Set grid lines to white
        axis.grid(color='white')
        
        # Set color for labels and ticks
        axis.xaxis.set_tick_params(color='white', labelcolor='white')
        axis.yaxis.set_tick_params(color='white', labelcolor='white')
        axis.spines['geo'].set_color('white')  # Set the outline color in Aitoff plot to white
        
        # Add colorbar with white labels and ticks
        cbar.set_label( colorbarLabel, color='white')
        cbar.ax.yaxis.set_tick_params( color='white' )
        cbar.outline.set_edgecolor( 'white' )  # Set colorbar outline to white
        
        # Set color of colorbar tick labels to white
        for label in cbar.ax.get_xticklabels():
            label.set_color( 'white' )

    if( saveto ):   fig.savefig( saveto, bbox_inches='tight' )
    
    # Show the plot
    pyplot.show()



