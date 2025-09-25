#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 12:32:31 2025

@author: rjedicke
"""


import numpy as np
import matplotlib.pyplot as pyplot

import PLOT_UTILITIES as plot
import UTILITIES as util

global gSMASS1
global gSMASS2



def load_smass1( smass1_file='/Users/rjedicke/Dropbox/data/smass1/smass1.all' ):
    """
    Reads fixed-width file with format:
      cols 0-3 : optional integer (4 chars)
      cols 5-12: designation (string, may include space, or be blank)
      cols 13-17: alt_id (integer)
      cols 19-..: value (float)
    """
    records = []
    with open( smass1_file, "r", encoding="utf-8") as f:
        
        for line in f:
            
            if not line.strip():
                continue  # skip empty lines

            # Fixed slices (Python is zero-based, end-exclusive)
            number_str    =        line[0:6  ].strip()
            designation   =        line[6:15 ].strip()
            wavelength_nm =   int( line[15:21].strip() ) / 10
            reflectance   = float( line[21:  ].strip() )

            # Parse values
            number = int(number_str) if number_str else None

            records.append( ( number, designation, wavelength_nm, reflectance ) )

    dtype = np.dtype([
        ("number",       object),    # object so we can hold None
        ("designation",   "U20"),    # Unicode string, up to 20 chars
        ("wavelength_nm", "i8"),
        ("reflectance",   "f8"),
    ])

    global gSMASS1
    
    gSMASS1 = np.array( records, dtype=dtype )
    
    return gSMASS1



def plot_smass1_spectrum( number_or_designation, xrange=(400,1100), yrange=(0.5,1.5) ):
    
    if(             len(gSMASS1[                 gSMASS1['number']     ==number_or_designation]) != 0 ):
        wavelength_nm = gSMASS1['wavelength_nm'][gSMASS1['number']     ==number_or_designation]
        reflectance   = gSMASS1['reflectance'][  gSMASS1['number']     ==number_or_designation]
        
    elif(           len(gSMASS1[                 gSMASS1['designation']==number_or_designation]) != 0 ):
        wavelength_nm = gSMASS1['wavelength_nm'][gSMASS1['designation']==number_or_designation]
        reflectance   = gSMASS1['reflectance'][  gSMASS1['designation']==number_or_designation]

    else:
        util.printWarning( 'SMASS1 number or designation not found' )
        return
    
    plot.plot2d( wavelength_nm, reflectance, xlabel='wavelength [nm]', xrange=xrange, ylabel='reflectance', yrange=yrange, linestyle='-', title=number_or_designation )
    
    return wavelength_nm, reflectance
    
    
    

def load_smass2( smass2_file='/Users/rjedicke/Dropbox/data/smass2/smass2_all_spfit.txt' ):
    
    global gSMASS2

    gSMASS2 = np.genfromtxt( smass2_file, names=('id','wavelenght_um','reflectance'), dtype=('U20','<f8','<f8') )
    
    return gSMASS2



def plot_smass2_spectrum( szID, xrange=(400,1100), yrange=(0.5,1.5) ):
    
    wavelength_nm = gSMASS2['wavelenght_um'][gSMASS2['id']==szID] * 1000
    reflectance   = gSMASS2['reflectance'][  gSMASS2['id']==szID]
   
    plot.plot2d( wavelength_nm, reflectance, xlabel='wavelength [nm]', xrange=xrange, ylabel='reflectance', yrange=yrange, linestyle='-', title=szID )
    
    return wavelength_nm, reflectance


    
def load_taxonomy_table( taxonomy_file='/Users/rjedicke/Dropbox/data/smass1/taxonomy.pds.table.txt' ):
    """
    Read the fixed-width taxonomy table described in the provided spec.

    Columns (with 1-based START_BYTE, BYTES):
      1) ASTEROID NUMBER                      (1, 5)   -> int or None if blank
      2) ASTEROID PROVISIONAL DESIGNATION     (7, 10)  -> str
      3) THOLEN CLASSIFICATION                (17, 6)  -> str
      4) THOLEN PARAMETERS                    (24, 3)  -> str
      5) BARUCCI CLASSIFICATION               (29, 2)  -> str
      6) BARUCCI PARAMETERS                   (32, 2)  -> str
      7) TEDESCO CLASSIFICATION               (37, 3)  -> str
      8) TEDESCO PARAMETERS                   (41, 2)  -> str
      9) HOWELL CLASSIFICATION                (46, 6)  -> str
     10) HOWELL PARAMETERS                    (53, 3)  -> str
     11) SMASS CLASSIFICATION                 (57, 3)  -> str
     12) COMMENT                               (61, 1)  -> str
    """
    # Helper to convert 1-based spec to Python slice and parse
    def slice_field(line, start_byte, nbytes, to_type='str'):
        # Convert to 0-based, end-exclusive
        start = start_byte - 1
        end = start + nbytes
        # Safe slice (line may be shorter)
        seg = line[start:end] if len(line) >= end else (line[start:] + ' ' * (end - len(line)))
        seg = seg.rstrip("\n")
        if to_type == 'int':
            s = seg.strip()
            return int(s) if s else None
        else:
            return seg.strip()

    records = []
    max_end = 61 - 1 + 1  # last column end (start 61, bytes 1) -> index 61
    with open(taxonomy_file, "r", encoding="utf-8") as f:
        for raw in f:
            # Skip blank/comment lines if present
            if not raw.strip() or raw.lstrip().startswith(("#", "OBJECT", "END_OBJECT")):
                continue
            # Ensure the line is long enough for slicing
            line = raw.rstrip("\n")
            if len(line) < max_end:
                line = line + " " * (max_end - len(line))

            rec = {
                "asteroid_number":         slice_field(line, 1, 5,  to_type='int'),
                "provisional":             slice_field(line, 7, 10),
                "tholen_class":            slice_field(line, 17, 6),
                "tholen_params":           slice_field(line, 24, 3),
                "barucci_class":           slice_field(line, 29, 2),
                "barucci_params":          slice_field(line, 32, 2),
                "tedesco_class":           slice_field(line, 37, 3),
                "tedesco_params":          slice_field(line, 41, 2),
                "howell_class":            slice_field(line, 46, 6),
                "howell_params":           slice_field(line, 53, 3),
                "smass_class":             slice_field(line, 57, 3),
                "comment":                 slice_field(line, 61, 1),
            }
            records.append(rec)

    return to_numpy( records )  # convert list of dicts to np array

# --- Optional: convert to pandas DataFrame or NumPy structured array ---

def to_pandas(records):
    import pandas as pd
    return pd.DataFrame.from_records(records)

def to_numpy(records):
    import numpy as np
    dtype = [
        ("asteroid_number", object),  # use object to allow None; change to 'i4' if you prefer -1 sentinel
        ("provisional", "U20"),
        ("tholen_class", "U6"),
        ("tholen_params", "U3"),
        ("barucci_class", "U2"),
        ("barucci_params", "U2"),
        ("tedesco_class", "U3"),
        ("tedesco_params", "U2"),
        ("howell_class", "U6"),
        ("howell_params", "U3"),
        ("smass_class", "U3"),
        ("comment", "U1"),
    ]
    arr = np.empty(len(records), dtype=dtype)
    for i, r in enumerate(records):
        arr[i] = tuple(r[k] for k, _ in dtype)
    return arr


def plot_all_smass2_spectra_of_type( szType, xrange=(400,1000), yrange=(0.5,1.5) ):
    
    load_smass2()
    
    taxonomy = load_taxonomy_table()
    
    StypesID = taxonomy['asteroid_number'][ taxonomy['tholen_class']==szType ]
    
    fig = axis = None
    
    for ID in StypesID:
        
        if( ID != None ):
    
            szID = '%06d' % (ID)
            
            wavelength_nm = gSMASS2['wavelenght_um'][gSMASS2['id']==szID] * 1000
            reflectance   = gSMASS2['reflectance'][  gSMASS2['id']==szID]
       
            fig, axis = plot.plot2d( wavelength_nm, reflectance, figure=fig, axis=axis, linestyle='-', title=szType+' type', bShow=False,
                                    xlabel='wavelength [nm]', xrange=xrange, ylabel='reflectance', yrange=yrange )
            
    pyplot.show()

    
    
    
#####################################################################################################################3

