#!/bin/bash
# -------------------------------------------------------------------------------- #
# Downloads the real-valued SuiteSparse matrices of Table 1 (Al Daas et al. 2023)
# used by Allrun.sh into <repo>/matrices. Run once on a G100 LOGIN node.
#   ./get_matrices.sh                      # default list (as in Allrun.sh)
#   ./get_matrices.sh Wissgott/parabolic_fem GHS_psdef/apache2   # others
# Not downloadable for this code: light_in_tissue, Chevron4 (complex valued).
# -------------------------------------------------------------------------------- #
LIST="Mulvey/finan512 Freescale/nxp1 Janna/CoupCons3D McRae/ecology2 CEMW/tmt_unsym
      Schmid/thermal2 Bourchtein/atmosmodj AMD/G3_circuit Janna/Transport
      Freescale/memchip Freescale/circuit5M_dc"
# other real matrices of Table 1: UTEP/Dubcova3 Williams/consph PARSEC/CO
#   Wissgott/parabolic_fem GHS_psdef/apache2 CEMW/tmt_sym

cd "$(dirname "$0")/../../matrices" || exit 1
[ $# -gt 0 ] && LIST="$*"
for gm in $LIST; do
    m=${gm#*/}
    if [ -f $m.mtx ]; then echo "$m: already here"; continue; fi
    echo "$m: downloading $gm"
    if wget -q --show-progress https://sparse.tamu.edu/MM/$gm.tar.gz -O $m.tar.gz &&
       tar xzf $m.tar.gz $m/$m.mtx && mv $m/$m.mtx . ; then
        echo "$m: ok ($(head -c 200 $m.mtx | head -n1))"
    else
        echo "$m: FAILED (check the group name on https://sparse.tamu.edu)"
    fi
    rm -rf $m $m.tar.gz
done
du -sh .
