from pysides.load_params import *
from pysides.gen_mass import *
from pysides.gen_sfr_props import *
from pysides.gen_magnification import *
from pysides.gen_fluxes import *
from pysides.gen_fluxes_filter import *
from pysides.gen_lines import *
from pysides.gen_outputs import *

params = load_params('PAR_FILES/SIDES_unclustered_1deg2_standard.par')

cat = gen_mass(params)

cat = gen_sfr_props(cat, params)

cat = gen_magnification(cat, params)

if __name__ == "__main__":  #added by SF to successfuly run the parallel computing

    cat = gen_fluxes(cat, params)

    cat = gen_fluxes_filter(cat, params)

    cat = gen_CO(cat, params)

    cat = gen_CII(cat, params)

    cat = gen_CI(cat, params)

    gen_outputs(cat, params)
