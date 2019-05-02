#!/bin/tcsh

foreach i (`seq 7 1 18`)
#foreach i (14 16 18)
    python Script_use_stftr.py  Subj$i EEG
end

# after swapping PPO10 and POO10, the number of time points in EEG was 109 for some subjects, 110 for others. I modified the code when copying mat data to epochs,
# for Subj 11 12 13, I set the max time to 109,  and for 14 16 18, max time = minimum of epoch times and data times
