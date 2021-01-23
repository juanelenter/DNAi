python generate_map.py
python generate_ped.py
./plink --ped ped.txt --map map.txt --noweb --no-fid --update-ids --no-sex --no-pheno --no-parents --recode --make-bed
