'''
This script uses histograms that are made using:
python scripts/histmakers/mz_dilepton.py -o output --axes etaAbsEta mll --postfix etaAbsEta_mll
'''

import os
import argparse
import h5py
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import mplhep as hep
from tqdm import tqdm
import hist

from wremnants.utilities.io_tools import base_io
from wremnants.postprocessing.datagroups.datagroups import Datagroups
from wums import ioutils, logging
from wums import boostHistHelpers as hh

import ROOT
from ROOT import RooFit, RooRealVar, RooArgList, RooFormulaVar
from ROOT import kRed, kBlue, kGreen, kMagenta

import random

parser = argparse.ArgumentParser(description='Computes muon curvature biases using the LHCb pseudomass method. Needs pseudomass histograms for Z->mumu.')
parser.add_argument('-i', '--input', type=str, help='Input file path', default="output/mz_dilepton_scetlib_dyturbo_CT18Z_N3p0LL_N2LO_Corr_etaAbsEta_mll_2016PostVFP.hdf5")
parser.add_argument('-o', '--outdir', type=str, help='Output directory path', default="output/")
parser.add_argument('--label', type=str, help='Result label', default="2016_postVFP")

# Enable multi-threading
ROOT.ROOT.EnableImplicitMT()

plt.figure() # Plotting an empty figure fixes incorrect style loading ?!'
plt.close()
plt.style.use(hep.style.CMS)
plt.rcParams["figure.facecolor"] = 'white' # needed to remove transparent background

def SetOwnerships(root_obj):
    """
    Gives pyROOT the ability to close the given ROOT objects when they become unused.
    Reclaiming all defined ROOT objects slows down the memory leakage.
    """
    if type(root_obj) is list:
        for i in range(len(root_obj)):
            ROOT.SetOwnership(root_obj[i], True)

    else:
        ROOT.SetOwnership(root_obj, True)

def make_root_hist(hist, hist_name="hist"):
    hist_TH1D = ROOT.TH1D(hist_name, "hist", len(hist.axes[0]), hist.axes[0].edges)
    for i in range(hist.axes[0].size):
        hist_TH1D.SetBinContent(i+1, hist[i].value)
        hist_TH1D.SetBinError(i+1, hist[i].variance**0.5)
    
    return hist_TH1D

def get_chi2_for_bin(hist1, hist2, model1, model2, x_var, hist1_name, hist2_name):
    rooHist1 = ROOT.RooDataHist("hist1_name", "dh", [x_var], hist1)
    rooHist2 = ROOT.RooDataHist("hist2_name", "dh", [x_var], hist2)
    
    test_chi2_1 = model1.createChi2(rooHist1, DataError="Expected", Verbose=True).getVal()
    test_chi2_2 = model2.createChi2(rooHist2, DataError="Expected", Verbose=True).getVal()

    return test_chi2_1 + test_chi2_2

def plot_bias_distributions(bias_arr, bias_err_arr, bin_centers_i, bin_centers_j,
                            save_fig=False,
                            save_paths=None,
                            title=None,
                            unit_label=None,
                            is_data=True,
                            bias_amplitude=None,
                            bias_err_amplitude=None,
                            bias_err_ratio_amplitude=None,
                            extend_colorbars=[None, None]):  

    bias_arr     = bias_arr.copy()
    bias_err_arr = bias_err_arr.copy()
    bias_err_arr[np.isclose(bias_err_arr, 0)] = np.nan
    if bias_amplitude is None:
        bias_amplitude = np.nanmax(np.abs(bias_arr))
    if bias_err_amplitude is None:
        bias_err_amplitude = np.nanmax(np.abs(bias_err_arr))
    if bias_err_ratio_amplitude is None:
        bias_err_ratio_amplitude = np.nanmax(np.abs(bias_arr/bias_err_arr))
    
    fig, ax = plt.subplots(3, 1, figsize=(11,18))
    hep.cms.label(loc=0, rlabel="", data=is_data, label="Preliminary", ax=ax[0])

    fig.suptitle(title)

    cmap1 = plt.get_cmap("bwr").copy()
    cmap1.set_extremes(under='magenta', over='yellow')
    cmap1.set_bad('black',1.)
    im1 = hep.hist2dplot((bias_arr, bin_centers_i, bin_centers_j), ax=ax[0], cmap=cmap1)
    im1.pcolormesh.set_clim(-bias_amplitude, bias_amplitude)
    ax[0].set_title("curvature bias", pad=40)
    ax[0].set_xlabel(r"$\eta$")
    ax[0].set_ylabel(r"$\phi$")
    im1.cbar.set_label(unit_label)
    if extend_colorbars[0] != None:
        im1.cbar.remove()
        cax1 = hep.append_axes(ax[0], size="7%", pad=0.2, position="right", extend=False)
        plt.colorbar(im1.pcolormesh, ax=ax[0], cax=cax1, extend=extend_colorbars[0], label=unit_label)

    cmap2 = plt.get_cmap("magma").copy()
    cmap2.set_extremes(under='magenta', over='lime')
    im2 = hep.hist2dplot((bias_err_arr, bin_centers_i, bin_centers_j), ax=ax[1], cmap=cmap2)
    im2.pcolormesh.set_clim(0, bias_err_amplitude)
    ax[1].set_title("std_error of bias", pad=20)
    ax[1].set_xlabel(r"$\eta$")
    ax[1].set_ylabel(r"$\phi$")
    im2.cbar.set_label(unit_label)
    if extend_colorbars[1] != None:
        im2.cbar.remove()
        cax2 = hep.append_axes(ax[1], size="7%", pad=0.2, position="right", extend=False)
        plt.colorbar(im2.pcolormesh, ax=ax[1], cax=cax2, extend=extend_colorbars[1], label=unit_label)

    cmap3 = plt.get_cmap("magma").copy()
    cmap3.set_extremes(under='magenta', over='lime')
    im3 = hep.hist2dplot((np.abs(bias_arr)/bias_err_arr, bin_centers_i, bin_centers_j), ax=ax[2], cmap=cmap3)
    im3.pcolormesh.set_clim(0, bias_err_ratio_amplitude)
    ax[2].set_title("|bias| / std_error", pad=20)
    ax[2].set_xlabel(r"$\eta$")
    ax[2].set_ylabel(r"$\phi$")
    if extend_colorbars[2] != None:
        im3.cbar.remove()
        cax3 = hep.append_axes(ax[2], size="7%", pad=0.2, position="right", extend=False)
        plt.colorbar(im3.pcolormesh, ax=ax[2], cax=cax3, extend=extend_colorbars[2])
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(save_paths[0], transparent=False)
        plt.clf()
    else:
        plt.show()
    
    # sigma difference between the curve fit and the bootstrap
    ratio_flattened = (bias_arr / bias_err_arr).flatten()
    ratio_flattened = ratio_flattened[~np.isnan(ratio_flattened)]
    bin_edges = np.histogram_bin_edges(ratio_flattened, bins="fd")
    x_arr = np.linspace(np.min(ratio_flattened), np.max(ratio_flattened), 1000)
    normal_dist_arr = (2*np.pi)**-0.5 * np.exp(-x_arr**2/2)
    normal_dist_arr *= len(bias_arr.flatten()) * (bin_edges[1]-bin_edges[0])
    
    plt.figure()
    hep.histplot(np.histogram((bias_arr / bias_err_arr).flatten(), density=False, bins=bin_edges), label="calculated", color="blue")
    plt.plot(x_arr, normal_dist_arr, lw=2, label="scaled normal\ndistribution\n"+r"$\sigma=1$", color="black", ls="--")
    plt.xlabel("|bias| / std_error")
    plt.ylabel("count / bin")
    if title == None:
        plt.title("Uncertainty-normalized distribution of bias", pad=20*2)
    else:
        plt.title(title + "\nUncertainty-normalized distribution of bias", pad=20*2)
    plt.legend(loc="upper right")
    hep.cms.label(loc=0, rlabel="", data=is_data, label="Preliminary")
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(save_paths[1], transparent=False)
        plt.clf()
    else:
        plt.show()

def plot_bias_comparison(bias_arr_1, bias_arr_2, bin_centers_i, bin_centers_j,
                         bias_err_arr_1, bias_err_arr_2,
                         title=None,
                         title1=None, title2=None, title3=None, title4=None,
                         cbar_label_bias=None,
                         cbar_label_pull=None,
                         save_fig=False,
                         save_path=None,
                         is_data=[True, True],
                         bias_amplitude=None,
                         bias_pull_amplitude=None,
                         extend_colorbars=[None, None]):

    """
    Plots two seperate 2D curvature bias arrays, their difference and optionally their difference divided by their total uncertainty.
    """

    bias_arr_1 = bias_arr_1.copy()
    bias_arr_2 = bias_arr_2.copy()
    if bias_amplitude is None:
        bias_amplitude = np.max([np.abs(bias_arr_1), np.abs(bias_arr_2), np.abs(bias_arr_1 - bias_arr_2)])
    
    fig, ax = plt.subplots(2, 2, figsize=(18,11))
    # hep.cms.label(loc=0, rlabel="", data=is_data[0], label="Preliminary", ax=ax[0,0])

    if title is not None: fig.suptitle(title)

    ax_label_size = 35
    tilte_pad = None

    cmap1 = plt.get_cmap("bwr").copy()
    cmap1.set_bad('black',1.)
    im1 = hep.hist2dplot((bias_arr_1, bin_centers_i, bin_centers_j), ax=ax[0,0], cmap=cmap1)
    im1.pcolormesh.set_clim(-bias_amplitude, bias_amplitude)
    ax[0,0].set_title(title1, pad=tilte_pad)
    ax[0,0].set_ylabel(r"$\phi$", fontsize=ax_label_size)
    im1.cbar.remove()
    
    cmap2 = plt.get_cmap("bwr").copy()
    # cmap2.set_extremes(under='magenta', over='yellow')
    im2 = hep.hist2dplot((bias_arr_2, bin_centers_i, bin_centers_j), ax=ax[0,1], cmap=cmap2)
    im2.pcolormesh.set_clim(-bias_amplitude, bias_amplitude)
    ax[0,1].set_title(title2, pad=tilte_pad)
    im2.cbar.set_label(cbar_label_bias)
    if extend_colorbars[1] != None:
        im2.cbar.remove()
        cax2 = hep.append_axes(ax[0,1], size="7%", pad=0.2, position="right", extend=False)
        plt.colorbar(im2.pcolormesh, ax=ax[0,1], cax=cax2, extend=extend_colorbars[0], label=cbar_label_bias)
    # plt.tight_layout()
    
    cmap3 = plt.get_cmap("bwr").copy()
    # cmap3.set_extremes(under='magenta', over='yellow')
    im3 = hep.hist2dplot((bias_arr_1 - bias_arr_2, bin_centers_i, bin_centers_j), ax=ax[1,0], cmap=cmap3)
    im3.pcolormesh.set_clim(-bias_amplitude, bias_amplitude)
    ax[1,0].set_title(title3, pad=tilte_pad)
    ax[1,0].set_xlabel(r"$\eta$", fontsize=ax_label_size)
    ax[1,0].set_ylabel(r"$\phi$", fontsize=ax_label_size)
    im3.cbar.remove()

    bias_total_err_arr = np.sqrt(bias_err_arr_1**2 + bias_err_arr_2**2)

    cmap4 = plt.get_cmap("magma").copy()
    cmap4.set_extremes(under='magenta', over='lime')
    im4 = hep.hist2dplot((np.abs(bias_arr_1 - bias_arr_2) / bias_total_err_arr, bin_centers_i, bin_centers_j), ax=ax[1,1], cmap=cmap4)
    im4.pcolormesh.set_clim(0, bias_pull_amplitude)
    ax[1,1].set_title(title4, pad=tilte_pad)
    ax[1,1].set_xlabel(r"$\eta$", fontsize=ax_label_size)
    # ax[1,1].set_ylabel(r"$\phi$", fontsize=ax_label_size)
    im4.cbar.set_label(cbar_label_pull)
    if extend_colorbars[1] != None:
        im4.cbar.remove()
        cax4 = hep.append_axes(ax[1,1], size="7%", pad=0.2, position="right", extend=False)
        plt.colorbar(im4.pcolormesh, ax=ax[1,1], cax=cax4, extend=extend_colorbars[1], label=cbar_label_pull)

    plt.tight_layout()

    
    if save_fig:
        plt.savefig(save_path, transparent=False, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def get_fit(hist1, hist2, draw):
    x = RooRealVar("x", "M^{+-}", 70, 120)
    
    mean     = RooRealVar("Mean",   "", 80, 100)
    assym    = RooRealVar("A",   "", -1, 1)
    mu_1     = RooFormulaVar("mu_1", "", "Mean*(1+A)", RooArgList(mean, assym))
    mu_2     = RooFormulaVar("mu_2", "", "Mean*(1-A)", RooArgList(mean, assym))
    
    sigmaL_1 = RooRealVar("SigmaL1", "", 1, 30)
    sigmaR_1 = RooRealVar("SigmaR1", "", 1, 30)
    alphaL_1 = RooRealVar("AlphaL1", "", 0.5, 5)
    alphaR_1 = RooRealVar("AlphaR1", "", 0.5, 5)
    nL_1     = RooRealVar("NL1",    "",  0.1, 200)
    nR_1     = RooRealVar("NR1",    "",  0.1, 200)
    
    mean.setVal(90)
    assym.setVal(0)
    sigmaL_1.setVal(5)
    sigmaR_1.setVal(5)
    alphaL_1.setVal(1)
    alphaR_1.setVal(1)
    nL_1.setVal(1)
    nR_1.setVal(1)
    
    crystalball_1 = ROOT.RooCrystalBall("signal1", "", x, mu_1, sigmaL_1, sigmaR_1, alphaL_1, nL_1, alphaR_1, nR_1)
    crystalball_2 = ROOT.RooCrystalBall("signal2", "", x, mu_2, sigmaL_1, sigmaR_1, alphaL_1, nL_1, alphaR_1, nR_1)
    
    category = ROOT.RooCategory("category", "")
    category.defineType("hist1")
    category.defineType("hist2")
    
    map1 = ROOT.std.map("std::string, TH1*")()
    map1.insert(("hist1", hist1))
    map1.insert(("hist2", hist2))
    
    combined_data = ROOT.RooDataHist("combined_data", "", RooArgList(x), category, map1)
    
    model1 = crystalball_1
    model2 = crystalball_2
    
    # Define a simultaneous model
    simultaneous_model = ROOT.RooSimultaneous("simultaneous_model", "", category)
    simultaneous_model.addPdf(model1, "hist1")
    simultaneous_model.addPdf(model2, "hist2")
    
    nll1 = simultaneous_model.createNLL(combined_data,
                                      # RooFit.Offset("initial"),
                                      )
    minimizer1 = ROOT.RooMinimizer(nll1)
    minimizer1.setMinimizerType("Minuit2")
    minimizer1.optimizeConst(True)
    minimizer1.setEps(1e-8)
    minimizer1.migrad()
    minimizer1.improve()
    # minimizer1.hesse()
    # minimizer1.minos()
    result1 = minimizer1.save()
    
    nll2 = simultaneous_model.createNLL(combined_data,
                                      RooFit.Offset("initial"),
                                      )
    minimizer2 = ROOT.RooMinimizer(nll2)
    minimizer2.setMinimizerType("Minuit2")
    minimizer2.optimizeConst(True)
    minimizer2.setEps(1e-8)
    minimizer2.migrad()
    minimizer2.improve()
    minimizer2.hesse()
    # minimizer2.minos()
    result2 = minimizer2.save()

    all_roofit_objects = [x, mean, assym, mu_1, mu_2, sigmaL_1, sigmaR_1, alphaL_1, alphaR_1, nL_1, nR_1,
                          crystalball_1, crystalball_2, category, map1, combined_data,
                          simultaneous_model, nll1, nll2, minimizer1, minimizer2, result1, result2]

    params = [alphaL_1, alphaR_1, assym, mean, nL_1, nR_1, sigmaL_1, sigmaR_1]
    param_values = [param.getVal() for param in params]
    
    val_assym   = assym.getValV()
    error_assym = assym.getError()

    hist1_name  = f"hist1"
    hist2_name  = f"hist2"
    chi2_val = get_chi2_for_bin(hist1, hist2, model1, model2, x, hist1_name, hist2_name)
    
    if draw:
        canvas = ROOT.TCanvas(f"canvas_separate_{np.random.randint(0, 100000)}", "Separate Fit", 1200, 750)
        
        frame1 = x.frame(ROOT.RooFit.Title("Histogram 1"))
        combined_data.plotOn(frame1, ROOT.RooFit.Cut("category==category::hist1"), MarkerColor=ROOT.kRed, LineColor=ROOT.kRed)
        simultaneous_model.plotOn(frame1, ROOT.RooFit.Slice(category, "hist1"), ROOT.RooFit.ProjWData(category, combined_data), LineColor=ROOT.kRed)
        
        frame2 = x.frame(ROOT.RooFit.Title("Histogram 2"))
        combined_data.plotOn(frame2, ROOT.RooFit.Cut("category==category::hist2"), MarkerColor=ROOT.kBlue, LineColor=ROOT.kBlue)
        simultaneous_model.plotOn(frame2, ROOT.RooFit.Slice(category, "hist2"), ROOT.RooFit.ProjWData(category, combined_data), LineColor=ROOT.kBlue)
        frame2.Draw()
        frame1.Draw("Same")

    SetOwnerships(all_roofit_objects)
    
    if draw:
        return val_assym, error_assym, param_values, chi2_val, canvas
    else:
        return val_assym, error_assym, param_values, chi2_val

def get_bootstrapped_std_error(centers, weights):
    test_data = np.stack([centers, weights], axis=0)
    def test_fun(*arr, axis):
        arr = np.array(arr)
        centers = arr[0,:]
        weights = arr[1,:]
        out = np.average(centers, weights=weights, axis=axis)
        return out

    result = stats.bootstrap(test_data, test_fun, vectorized=True, axis=0, paired=True, method="BCa")
    return result.standard_error

def compute_fit_arrays(hist_Mplus, hist_Mminus, hist_Pplus_rec, hist_Pminus_rec, make_plots=False):
    shape_hist = hist_Mplus.shape[:2]
    
    muPlus_pmag_reciprocal_mean_arr  = np.zeros(shape_hist)
    muMinus_pmag_reciprocal_mean_arr = np.zeros(shape_hist)
    Pplus_rec_error_arr              = np.zeros(shape_hist)
    Pminus_rec_error_arr             = np.zeros(shape_hist)
    assym_arr_separate_fit           = np.zeros(shape_hist)
    assym_error_arr_separate_fit     = np.zeros(shape_hist)
    chi2_arr_separate_fit            = np.zeros(shape_hist)
    
    curve_param_arr_separate_fit = np.zeros([*shape_hist, 6]) # sigmaL, sigmaR, alphaL, alphaR, nL, nR

    if make_plots:
        canvas_arr = np.empty(shape_hist, dtype=object)
    
    for i in tqdm(range(0, shape_hist[0])):
        param_values_i = []
        hist_12_ij_list_separate = []
        for j in range(0, shape_hist[1]):
            subfit_label = f"{i}_{j}"
            
            hist1_ij = make_root_hist(hist_Mplus [i,j,:])
            hist2_ij = make_root_hist(hist_Mminus[i,j,:])
    
            hist1_ij.SetName(f"hist1_{subfit_label}")
            hist2_ij.SetName(f"hist2_{subfit_label}")
            
            if make_plots:
                assym_current, assym_current_err, param_values_current, chi2_val, canvas_current = get_fit(hist1_ij, hist2_ij, draw=True)
                canvas_arr[i,j] = canvas_current
            else:
                assym_current, assym_current_err, param_values_current, chi2_val = get_fit(hist1_ij, hist2_ij, draw=False)
    
            assym_arr_separate_fit[i,j]         = assym_current
            assym_error_arr_separate_fit[i,j]   = assym_current_err
            curve_param_arr_separate_fit[i,j,0] = param_values_current[0]
            curve_param_arr_separate_fit[i,j,1] = param_values_current[1]
            curve_param_arr_separate_fit[i,j,2] = param_values_current[4]
            curve_param_arr_separate_fit[i,j,3] = param_values_current[5]
            curve_param_arr_separate_fit[i,j,4] = param_values_current[6]
            curve_param_arr_separate_fit[i,j,5] = param_values_current[7]
    
            chi2_arr_separate_fit[i,j] = chi2_val

            muPlus_pmag_reciprocal_mean_arr [i,j] = np.average(hist_Pplus_rec [i,j,:].axes[0].centers, weights=hist_Pplus_rec [i,j,:].values())
            muMinus_pmag_reciprocal_mean_arr[i,j] = np.average(hist_Pminus_rec[i,j,:].axes[0].centers, weights=hist_Pminus_rec[i,j,:].values())

            Pplus_rec_error_arr [i,j] = get_bootstrapped_std_error(hist_Pplus_rec [i,j,:].axes[0].centers, weights=hist_Pplus_rec [i,j,:].values())
            Pminus_rec_error_arr[i,j] = get_bootstrapped_std_error(hist_Pminus_rec[i,j,:].axes[0].centers, weights=hist_Pminus_rec[i,j,:].values())
    
    bias_arr_separate_fit         = -assym_arr_separate_fit * 1/2 * (muPlus_pmag_reciprocal_mean_arr + muMinus_pmag_reciprocal_mean_arr)
    bias_error_arr_separate_fit_1 = assym_error_arr_separate_fit * 1/2 * (muPlus_pmag_reciprocal_mean_arr + muMinus_pmag_reciprocal_mean_arr)
    bias_error_arr_separate_fit_2 = np.abs(assym_arr_separate_fit * 1/2 * Pplus_rec_error_arr)
    bias_error_arr_separate_fit_3 = np.abs(assym_arr_separate_fit * 1/2 * Pminus_rec_error_arr)
    # Correlations between the errors are expected to be negligible
    bias_error_arr_separate_fit   = (bias_error_arr_separate_fit_1**2 + bias_error_arr_separate_fit_2**2 + bias_error_arr_separate_fit_3**2)**0.5

    if make_plots:
        return bias_arr_separate_fit, bias_error_arr_separate_fit, chi2_arr_separate_fit, curve_param_arr_separate_fit, canvas_arr
    else:
        return bias_arr_separate_fit, bias_error_arr_separate_fit, chi2_arr_separate_fit, curve_param_arr_separate_fit

args = parser.parse_args()

h5_path = args.input
out_dir_path = args.outdir
result_label = args.label

if not os.path.exists(out_dir_path):
    os.makedirs(out_dir_path)
    
plot_paths_data       = [f"{out_dir_path}/muon_curvature_bias_data.png",
                         f"{out_dir_path}/uncertainty_normalized_distribution_bias_data.png"]
plot_paths_mc         = [f"{out_dir_path}/muon_curvature_bias_mc.png",
                         f"{out_dir_path}/uncertainty_normalized_distribution_bias_mc.png"]
plot_path_comparison =   f"{out_dir_path}/muon_curvature_bias_comparison.png"
h5_file = h5py.File(h5_path, "r")
results = base_io.load_results_h5py(h5_file)

groups = Datagroups(h5_path)
groups.loadHistsForDatagroups("nominal_etaPlus_pseudomass_phiPlus_pseudomass_pseudomassPlus", syst="")
groups.loadHistsForDatagroups("nominal_etaMinus_pseudomass_phiMinus_pseudomass_pseudomassMinus", syst="")
groups.loadHistsForDatagroups("nominal_etaPlus_pseudomass_phiPlus_pseudomass_ptPlus_reciprocal", syst="")
groups.loadHistsForDatagroups("nominal_etaMinus_pseudomass_phiMinus_pseudomass_ptMinus_reciprocal", syst="")

data_hist_Mplus = None
data_hist_Mminus = None
data_hist_Pplus_rec = None
data_hist_Pminus_rec = None


mc_hists_Mplus = []
mc_hists_Mminus = []
mc_hists_Pplus_rec = []
mc_hists_Pminus_rec = []

mc_hist_sum_Mplus = None
mc_hist_sum_Mminus = None
mc_hist_sum_Pplus_rec = None
mc_hist_sum_Pminus_rec = None

for group_name in groups.groups.keys():
    group = groups.groups[group_name]
    print(group.name)
    for group_member in group.members:
        print(group_member.name, group_member.is_data)

    if group_name == "Data":
        data_hist_Mplus  = group.hists['nominal_etaPlus_pseudomass_phiPlus_pseudomass_pseudomassPlus']
        data_hist_Mminus = group.hists['nominal_etaMinus_pseudomass_phiMinus_pseudomass_pseudomassMinus']
        data_hist_Pplus_rec  = group.hists['nominal_etaPlus_pseudomass_phiPlus_pseudomass_ptPlus_reciprocal']
        data_hist_Pminus_rec = group.hists['nominal_etaMinus_pseudomass_phiMinus_pseudomass_ptMinus_reciprocal']
        
    else:
        mc_hists_Mplus .append(group.hists['nominal_etaPlus_pseudomass_phiPlus_pseudomass_pseudomassPlus'])
        mc_hists_Mminus.append(group.hists['nominal_etaMinus_pseudomass_phiMinus_pseudomass_pseudomassMinus'])
        mc_hists_Pplus_rec .append(group.hists['nominal_etaPlus_pseudomass_phiPlus_pseudomass_ptPlus_reciprocal'])
        mc_hists_Pminus_rec.append(group.hists['nominal_etaMinus_pseudomass_phiMinus_pseudomass_ptMinus_reciprocal'])

        if mc_hist_sum_Mplus is not None:
            mc_hist_sum_Mplus  = hh.addHists(mc_hist_sum_Mplus , group.hists['nominal_etaPlus_pseudomass_phiPlus_pseudomass_pseudomassPlus'])
            mc_hist_sum_Mminus = hh.addHists(mc_hist_sum_Mminus, group.hists['nominal_etaMinus_pseudomass_phiMinus_pseudomass_pseudomassMinus'])
            mc_hist_Pplus_rec  = hh.addHists(mc_hist_sum_Pplus_rec , group.hists['nominal_etaPlus_pseudomass_phiPlus_pseudomass_ptPlus_reciprocal'])
            mc_hist_Pminus_rec = hh.addHists(mc_hist_sum_Pminus_rec, group.hists['nominal_etaMinus_pseudomass_phiMinus_pseudomass_ptMinus_reciprocal'])
        else:
            mc_hist_sum_Mplus      = group.hists['nominal_etaPlus_pseudomass_phiPlus_pseudomass_pseudomassPlus']
            mc_hist_sum_Mminus     = group.hists['nominal_etaMinus_pseudomass_phiMinus_pseudomass_pseudomassMinus']
            mc_hist_sum_Pplus_rec  = group.hists['nominal_etaPlus_pseudomass_phiPlus_pseudomass_ptPlus_reciprocal']
            mc_hist_sum_Pminus_rec = group.hists['nominal_etaMinus_pseudomass_phiMinus_pseudomass_ptMinus_reciprocal']
    
data_hist_Mplus        = data_hist_Mplus       [::4j,::4j,:]
data_hist_Mminus       = data_hist_Mminus      [::4j,::4j,:]
data_hist_Pplus_rec    = data_hist_Pplus_rec   [::4j,::4j,:]
data_hist_Pminus_rec   = data_hist_Pminus_rec  [::4j,::4j,:]
mc_hist_sum_Mplus      = mc_hist_sum_Mplus     [::4j,::4j,:]
mc_hist_sum_Mminus     = mc_hist_sum_Mminus    [::4j,::4j,:]
mc_hist_sum_Pplus_rec  = mc_hist_sum_Pplus_rec [::4j,::4j,:]
mc_hist_sum_Pminus_rec = mc_hist_sum_Pminus_rec[::4j,::4j,:]

bias_arr_data, bias_error_arr_data, chi2_arr_data, curve_param_arr_data = compute_fit_arrays(data_hist_Mplus, data_hist_Mminus, data_hist_Pplus_rec, data_hist_Pminus_rec)
bias_arr_mc_sum, bias_error_arr_mc_sum, chi2_arr_mc_sum, curve_param_arr_mc_sum = compute_fit_arrays(mc_hist_sum_Mplus, mc_hist_sum_Mminus, mc_hist_sum_Pplus_rec, mc_hist_sum_Pminus_rec)


plot_bias_distributions(bias_arr_data, bias_error_arr_data, data_hist_Mplus.axes[0].edges, data_hist_Mplus.axes[1].edges, is_data=True, save_fig=True, save_paths=plot_paths_data,
                        bias_amplitude=None, bias_err_amplitude=None, extend_colorbars=[None, None, None])

plot_bias_distributions(bias_arr_mc_sum, bias_error_arr_mc_sum, data_hist_Mplus.axes[0].edges, data_hist_Mplus.axes[1].edges, is_data=False, save_fig=True, save_paths=plot_paths_mc,
                        bias_amplitude=None, bias_err_amplitude=None, extend_colorbars=[None, None, None])

plot_bias_comparison(bias_arr_data, bias_arr_mc_sum, data_hist_Mplus.axes[0].edges, data_hist_Mplus.axes[1].edges,
                     bias_error_arr_data, bias_error_arr_mc_sum,
                     title=result_label,
                     title1="Data", title2="MC", title3="Difference", title4="Pull of difference",
                     cbar_label_bias="1/GeV",
                     save_fig=True, save_path=plot_path_comparison,
                     bias_amplitude=None, bias_pull_amplitude=None, extend_colorbars=[None, None], )

bias_weights = data_hist_Mplus[:,:,::hist.sum].to_numpy()[0] + data_hist_Mminus[:,:,::hist.sum].to_numpy()[0]

bias_difference       = bias_arr_data - bias_arr_mc_sum
bias_difference_error = (bias_error_arr_data**2 + bias_error_arr_mc_sum**2)**0.5

mean_bias_difference       = np.average(bias_difference, weights=bias_weights)
mean_bias_difference_error = np.sum(bias_weights**2 * bias_difference_error**2)**0.5 / np.sum(bias_weights)

print(f"Weighted mean of the bias difference: {mean_bias_difference:.04g} +- {mean_bias_difference_error:.04g} 1/GeV")