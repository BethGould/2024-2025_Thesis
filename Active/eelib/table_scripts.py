# table_scripts.py

# Author: Elizabeth Gould
# Date Last Edit: 15.03.2026

# This file contains a bunch of functions used to prune my datasets from the grid_BVP class.

# clean_table(tbl) cleans the table produced by pd.DataFrame(grid.derivs) by grid runs of grid_BVP.
# clean_table_MC(tbl) cleans this table for Monte Carlo runs of grid_BVP.
# Both of these return a new table, which is a new DataFrame object and not a splice of the old table.

# find_ind_var(i, R, B, dk, mu) returns an array of strings which contains the names of the arrays 
#                               R[i], B[i], dk[i], and mu[i] that have more than one value.


# Libraries
import numpy as np
import pandas as pd
import warnings

# Cleans the table produced by pd.DataFrame(grid.derivs) by grid runs of grid_BVP, 
# and returns a new DataFrame object with the reduced dataset.
# Note that this technically works with the Monte Carlo variants, but it is too slow to be effective.
def clean_table(tbl):
    val_list = [] # Create a list, in which to store all the rows we wish to keep.
    R_arr = tbl["R"].unique()
    for rr in R_arr:
        B_arr = tbl["B"].unique()
        for bb in B_arr:
            dk_arr = tbl['dk'].unique()
            for kk in dk_arr:
                mu_arr = tbl['mu'].unique()
                for mm in mu_arr:
                    # For each such sets of parameters, select all rows which share this set of parameters.
                    tbl_select = tbl[(tbl["R"] == rr)&(tbl["B"] == bb)&(tbl["dk"] == kk)&(tbl["mu"]==mm)]
                    # Within the try statement is code to select the best of the rows and add it to my 
                    # val_list list. It is contained in a try statement so that if there are no such rows,
                    # it will just ignore this and not add anything to the list.
                    try:
                        ll = np.min(list(tbl_select["A max new"])) # Select the row with the lowest max amplitude.
                        tbl_row = tbl_select[tbl_select["A max new"]==ll].values.tolist() # Extract the data from the row.
                        val_list.append(tbl_row[0]) # And add it to my list of saved rows.
                    except ValueError:
                        pass # Add nothing and continue to the next parameter set if there are no values with the given parameters.

    # A list of the names of my columns to use when creating a DataFrame.
    col_names = ["R", "B", "dk", "mu", 'dpsi0', 'a0', 'b0', 'A max 0', 'I0', 'dpsi', 'a', 'b', 'A max new', 'I v2', 'effective mu']

    # Now create a new DataFrame with my desired rows and the given column names.
    # A list of equal sized lists is another valid data format for creating a DataFrame.
    new_tbl = pd.DataFrame(val_list, columns=col_names)

    # Recast all of our real values as real numbers, as due to the existance of complex values for some 
    # columns, the real column had been recast as complex numbers when extracting the data.
    # Recasting complex numbers as real numbers gives a warning, but the numbers being recast here
    # were originally real numbers which had been recast as complex, so I can safely ignore this warning.
    with warnings.catch_warnings(action="ignore"):
        new_tbl['R'] = new_tbl['R'].astype(float)
        new_tbl['B'] = new_tbl['B'].astype(float)
        new_tbl['dk'] = new_tbl['dk'].astype(float)
        new_tbl['mu'] = new_tbl['mu'].astype(float)
        new_tbl['A max 0'] = new_tbl['A max 0'].astype(float)
        new_tbl['I0'] = new_tbl['I0'].astype(float)
        new_tbl['A max new'] = new_tbl['A max new'].astype(float)
        new_tbl['I v2'] = new_tbl['I v2'].astype(float)
        new_tbl['effective mu'] = new_tbl['effective mu'].astype(float)

    # Return the resulting DataFrame. Our result is a new DataFrame, not a splice (which will be an important fact later).
    return new_tbl

# This is the same script, but in a form which takes account of the fact that Monte Carlo algorithms will 
# have all new parameters. I will only examine one parameter, selecting all rows with equal values for that 
# parameter, and saving the best of them. If applied to a grid of data, this algorithm will throw out the
# grid, but for Monte Carlo datasets, the chance of loosing a desired data point is very small.
def clean_table_MC(tbl):
    val_list = [] # Create a list, in which to store all the rows we wish to keep.
    R_arr = tbl["R"].unique()
    for rr in R_arr:
        # For each such sets of parameters, select all rows which share this set of parameters.
        tbl_select = tbl[tbl["R"] == rr] 
        ll = np.min(list(tbl_select["A max new"])) # Select the row with the lowest max amplitude.
        tbl_row = tbl_select[tbl_select["A max new"]==ll].values.tolist() # Extract the data from the row.
        val_list.append(tbl_row[0]) # And add it to my list of saved rows.

    # A list of the names of my columns to use when creating a DataFrame.
    col_names = ["R", "B", "dk", "mu", 'dpsi0', 'a0', 'b0', 'A max 0', 'I0', 'dpsi', 'a', 'b', 'A max new', 'I v2', 'effective mu']#, columns = col_names
    
    # Now create a new DataFrame with my desired rows and the given column names.
    # A list of equal sized lists is another valid data format for creating a DataFrame.
    new_tbl = pd.DataFrame(val_list, columns=col_names)

    # Recast all of our real values as real numbers, as due to the existance of complex values for some 
    # columns, the real column had been recast as complex numbers when extracting the data.
    # Recasting complex numbers as real numbers gives a warning, but the numbers being recast here
    # were originally real numbers which had been recast as complex, so I can safely ignore this warning.
    with warnings.catch_warnings(action="ignore"):
        new_tbl['R'] = new_tbl['R'].astype(float)
        new_tbl['B'] = new_tbl['B'].astype(float)
        new_tbl['dk'] = new_tbl['dk'].astype(float)
        new_tbl['mu'] = new_tbl['mu'].astype(float)
        new_tbl['A max 0'] = new_tbl['A max 0'].astype(float)
        new_tbl['I0'] = new_tbl['I0'].astype(float)
        new_tbl['A max new'] = new_tbl['A max new'].astype(float)
        new_tbl['I v2'] = new_tbl['I v2'].astype(float)
        new_tbl['effective mu'] = new_tbl['effective mu'].astype(float)
    
    # Return the resulting DataFrame. Our result is a new DataFrame, not a splice (which will be an important fact later).
    return new_tbl

# This code is used to remove the parameters which are constant, thus avoiding the 
# issue of plotting variables with no variance. It creates a list of our independent
# parameters where there are multiple values on the table.
def find_ind_var(i, R, B, dk, mu):
    R_len = R[i].shape[0]
    B_len = B[i].shape[0]
    dk_len = dk[i].shape[0]
    mu_len = mu[i].shape[0]
    vars  = []
    if R_len > 1:
        vars.append("R")
    if B_len > 1:
        vars.append("B")
    if dk_len > 1:
        vars.append("dk")
    if mu_len > 1:
        vars.append("mu")
    return vars