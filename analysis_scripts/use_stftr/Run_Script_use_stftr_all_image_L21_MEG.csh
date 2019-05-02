#!/bin/tcsh

foreach i (`seq 1 1 18`)
    python Script_use_stftr.py  Subj$i MEG
end
