#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 18:00:18 2020

@author: ignaciohounie
"""
SNP_num = 60671 # number of SNPs, could be calculated from file to convert (TO-DO)
f = open("map.txt", "w")
for i in range(SNP_num):
  line = "1  "+ str(i)+ "  0  0"
  f.write(line)
  f.write("\n")
f.close()