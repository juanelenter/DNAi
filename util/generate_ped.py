#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 18:00:18 2020

@author: ignaciohounie
"""

#########################################################
# LUT to translate allele frequency to bynary encoding
#######################################################
cero = '1 1'
uno = '1 2'
dos = '2 2'
na = '0 0'

########################################################
# input and output file path
#######################################################
input_pth = "snps.txt"
output_pth = "ped.txt"

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
        # converting symbols using LUT
        spaced= spaced.replace('0', 'c' )
        spaced = spaced.replace('1', 'u')
        spaced = spaced.replace('2', 'd')
        spaced = spaced.replace('3', 'n')
        spaced = spaced.replace('4', 'n')
        spaced = spaced.replace('5', 'n')
        spaced= spaced.replace('c', cero )
        spaced = spaced.replace('u', uno)
        spaced = spaced.replace('d', dos)
        spaced = spaced.replace('n', na)
        # Adding individual id
        spaced = str(idx)+ " "+  spaced
        output.write(spaced)
output.close()
