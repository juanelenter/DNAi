#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 17:30:05 2020

@author: ignaciohounie
"""



########################################################
# input and output file path
#######################################################
input_pth = "snps.txt"
output_pth = "snps_idless.txt"

########################################################
# Read input file, convert to ped format, and write to output
#########################################################
output = open(output_pth, "w")
with open(input_pth, "r") as snps:
    for idx, line in enumerate(snps):
        # Cow identifier ignored
        id_less = line.split(' ', 1)[1]
        # Adding spaces
        spaced = " ".join(id_less)
        output.write(spaced)
output.close()