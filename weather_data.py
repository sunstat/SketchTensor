
# coding: utf-8

# In[1]:


import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt 
import pickle
import tensorly
import matplotlib.ticker as ticker
from simulation_realdata import *


def simrun_name(name, inv_factor): 
    return "data/"+name+"_frk"+str(inv_factor)+".pickle"
def simplot_name(name, inv_factor): 
    return "plots/"+name+"_frk"+str(inv_factor)+".pdf" 

def run_realdata_frk(data, inv_factor, name, random_seed = 1): 
    ranks = (np.array(data.shape)/inv_factor).astype(int)    
    _, ho_svd_rerr = tucker_data(data, ranks) 
    tucker_result = np.repeat(ho_svd_rerr, len(np.arange(2/inv_factor,2/5, (2/5 -1/inv_factor)/10))).tolist()
    one_pass_result = []
    two_pass_result = []
    for factor in np.arange(2/inv_factor,2/5, (2/5 -1/inv_factor)/10):  
        ks = (np.array(data.shape)*factor).astype(int)
        ss = 2*ks+1 
        _, two_pass_rerr = two_pass_data(data, ranks, ks)
        _, one_pass_rerr = one_pass_data(data, ranks, ks, ss) 
        one_pass_result.append(one_pass_rerr)
        two_pass_result.append(two_pass_rerr)
    result = [tucker_result, two_pass_result, one_pass_result] 
    pickle.dump( result, open(simrun_name(name,inv_factor), "wb" ) )
    return result

def plot_realdata_frk(data, inv_factor, name, fontsize = 18): 
    ranks = (np.array(data.shape)/inv_factor).astype(int)    
    kratio = np.arange(2/inv_factor,2/5, (2/5 -1/inv_factor)/10)
    result = pickle.load(open(simrun_name(name, inv_factor), "rb" ) )
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    plt.rc('text', usetex=True)    
    plt.figure(figsize=(6,5))
    plt.plot(kratio,result[0], label = 'HOOI', linestyle = ':', marker = 'o', markeredgewidth=1,markeredgecolor='g', markerfacecolor='None') 
    plt.plot(kratio,result[1], label = 'Two Pass',linestyle = '--',marker = 'X')
    plt.plot(kratio,result[2], label = 'One Pass', marker = 's',markeredgewidth=1,markeredgecolor='orange', markerfacecolor='None') 
    plt.legend(loc = 'best')
    plt.title("rk/I = %s"%round(1/inv_factor,3))
    plt.xlabel('k/I')
    plt.ylabel('Relative Error')
    plt.yscale('log')
    plt.minorticks_off()
    alldata = np.concatenate(result)
    ymin = min(alldata)
    ymax = max(alldata) 
    def round_to_n(x,n): 
        return round(x,-int(np.floor(np.log10(abs(x))))+n-1) 
    ticks = [round_to_n(i,3) for i in np.arange(ymin, ymax+(ymax-ymin)/5,(ymax-ymin)/5)] 
    plt.yticks(ticks)
    plt.axes().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e')) 
    plt.axes().title.set_fontsize(fontsize)
    plt.axes().xaxis.label.set_fontsize(fontsize)
    plt.axes().yaxis.label.set_fontsize(fontsize)
    plt.rc('legend',fontsize = fontsize)
    plt.rc('xtick', labelsize = fontsize) 
    plt.rc('ytick', labelsize = fontsize) 
    plt.tight_layout()
    plt.savefig(simplot_name(name,inv_factor))
    plt.show()


if __name__ == '__main__':   
    PSL = nc.Dataset("data/b.e11.B1850C5CN.f09_g16.005.cam.h0.PSL.200001-209912.nc").variables['PSL'][:] 
    CLOUD = nc.Dataset("data/b.e11.BRCP85C5CNBDRD.f09_g16.013.cam.h0.CLOUD.208101-210012.nc").variables['CLOUD'][:]
    run_realdata_frk(CLOUD,15,"CLOUD")
    run_realdata_frk(CLOUD,10,"CLOUD")
    run_realdata_frk(CLOUD,8,"CLOUD")
    run_realdata_frk(PSL,15,"PSL")
    run_realdata_frk(PSL,10,"PSL")
    run_realdata_frk(PSL,8,"PSL")

