import pandas as pd

import pyPPG

from pyPPG.ppg_bm.ppg_sig import get_ppg_sig, BIOMARKERS_LST as _LST_PPG_SIG
from pyPPG.ppg_bm.sig_ratios import get_sig_ratios, BIOMARKERS_LST as _LST_SIG_RATIOS
from pyPPG.ppg_bm.ppg_derivs import get_ppg_derivs, BIOMARKERS_LST as _LST_PPG_DERIVS
from pyPPG.ppg_bm.derivs_ratios import get_derivs_ratios, BIOMARKERS_LST as _LST_DERIVS_RATIOS
from pyPPG.ppg_bm.bm_extraction import get_biomarkers as _bm_extract
from pyPPG.ppg_bm.statistics import get_statistics

class BmCollection:

    ###########################################################################
    ######################## Initialization of Biomarkers #####################
    ###########################################################################
    def __init__(self, s: pyPPG.PPG, fp: pyPPG.Fiducials):
        """
        The purpose of the Biomarkers class is to calculate the PPG biomarkers.

        :param s: object of PPG signal
        :type s: pyPPG.PPG object
        :param fp: object of fiducial points
        :type fp: pyPPG.Fiducials object

        """

        self.s = s
        self.fp = fp

    ###########################################################################
    ############################ Get PPG Biomarkers ###########################
    ###########################################################################
    def get_biomarkers (self, get_stat=True):
        """
        This function retrieves the list of biomarkers, computes their values, and calculates associated statistics.

        :param get_stat: a bool for calculating the statistics of biomarkers
        :type get_stat: bool

        :return:
            - bm_defs: dictionary of biomarkers with name, definition and unit
            - bm_vals: dictionary of biomarkers with values
            - bm_stats: dictionary of biomarkers with statistics
        """

        s=self.s
        fp = self.fp

        ## Get Biomarkers — single pass over beats.
        # The four sibling get_X helpers each ran their own per-beat loop and
        # rebuilt BmExctator from scratch every beat, doing the per-beat
        # fiducial decomposition (1st/2nd/3rd-derivative landmark detection)
        # four redundant times. Concatenate the four definition tables and
        # invoke the extractor once with the union; then slice the result
        # into the four category frames by column name.
        all_lst = pd.concat(
            [_LST_PPG_SIG, _LST_SIG_RATIOS, _LST_PPG_DERIVS, _LST_DERIVS_RATIOS],
            ignore_index=True,
        )
        pw, bm_all = _bm_extract(s, fp, all_lst.name)

        bm_ppg_sig       = bm_all[_LST_PPG_SIG.name.tolist()]
        bm_sig_ratios    = bm_all[_LST_SIG_RATIOS.name.tolist()]
        bm_ppg_derivs    = bm_all[_LST_PPG_DERIVS.name.tolist()]
        bm_derivs_ratios = bm_all[_LST_DERIVS_RATIOS.name.tolist()]

        bm_vals={'ppg_sig': bm_ppg_sig , 'sig_ratios': bm_sig_ratios, 'ppg_derivs': bm_ppg_derivs, 'derivs_ratios': bm_derivs_ratios}
        bm_defs = {'ppg_sig': _LST_PPG_SIG, 'sig_ratios': _LST_SIG_RATIOS, 'ppg_derivs': _LST_PPG_DERIVS, 'derivs_ratios': _LST_DERIVS_RATIOS}

        ## Get Statistics
        if get_stat:
            bm_stats = get_statistics(fp.sp, fp.on, bm_vals)
        else:
            bm_stats={'ppg_sig': [], 'sig_ratios': [], 'ppg_derivs': [], 'derivs_ratios': []}

        ## Update index names
        BM_keys = bm_vals.keys()
        for key in BM_keys:
            bm_vals[key] = bm_vals[key].rename_axis('Index of pulse')
            bm_vals[key].insert(0,'TimeStamp',pw.onset)
            bm_defs[key] = bm_defs[key].rename_axis('No. biomarkers')
            if get_stat: bm_stats[key] = bm_stats[key].rename_axis('Statistics')

        if get_stat:
            return bm_defs, bm_vals, bm_stats
        else:
            return bm_defs, bm_vals
