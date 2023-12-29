#!/bin/sh

source activate fastvgs

M="DINO"
G="exp6M"
ROOT="/worktmp/khorrami/current"
MROOT="/worktmp/khorrami/current/FaST/experiments"/$M/$G

mkdir $ROOT/"semtest/S"/$M/$G


MNAME="expS0"

SNAME=$M/$G/"S0_aL_vO"
AF="COCO"
VF="images/original"
python semDINO.py --exp_dir $MROOT/$MNAME --root $ROOT --Sname $SNAME --afiles $AF --vfiles $VF

SNAME=$M/$G/"S0_aL_vM"
AF="COCO"
VF="images/masked"
python semDINO.py --exp_dir $MROOT/$MNAME --root $ROOT --Sname $SNAME --afiles $AF --vfiles $VF

SNAME=$M/$G/"S0_aL_vB"
AF="COCO"
VF="images/blurred"
python semDINO.py --exp_dir $MROOT/$MNAME --root $ROOT --Sname $SNAME --afiles $AF --vfiles $VF



MNAME="expS1"

SNAME=$M/$G/"S1_aL_vO"
AF="COCO"
VF="images/original"
python semDINO.py --exp_dir $MROOT/$MNAME --root $ROOT --Sname $SNAME --afiles $AF --vfiles $VF

SNAME=$M/$G/"S1_aL_vM"
AF="COCO"
VF="images/masked"
python semDINO.py --exp_dir $MROOT/$MNAME --root $ROOT --Sname $SNAME --afiles $AF --vfiles $VF

SNAME=$M/$G/"S1_aL_vB"
AF="COCO"
VF="images/blurred"
python semDINO.py --exp_dir $MROOT/$MNAME --root $ROOT --Sname $SNAME --afiles $AF --vfiles $VF

MNAME="expS2"

SNAME=$M/$G/"S2_aL_vO"
AF="COCO"
VF="images/original"
python semDINO.py --exp_dir $MROOT/$MNAME --root $ROOT --Sname $SNAME --afiles $AF --vfiles $VF

SNAME=$M/$G/"S2_aL_vM"
AF="COCO"
VF="images/masked"
python semDINO.py --exp_dir $MROOT/$MNAME --root $ROOT --Sname $SNAME --afiles $AF --vfiles $VF

SNAME=$M/$G/"S2_aL_vB"
AF="COCO"
VF="images/blurred"
python semDINO.py --exp_dir $MROOT/$MNAME --root $ROOT --Sname $SNAME --afiles $AF --vfiles $VF


MNAME="expS3"

SNAME=$M/$G/"S3_aL_vO"
AF="COCO"
VF="images/original"
python semDINO.py --exp_dir $MROOT/$MNAME --root $ROOT --Sname $SNAME --afiles $AF --vfiles $VF

SNAME=$M/$G/"S3_aL_vM"
AF="COCO"
VF="images/masked"
python semDINO.py --exp_dir $MROOT/$MNAME --root $ROOT --Sname $SNAME --afiles $AF --vfiles $VF

SNAME=$M/$G/"S3_aL_vB"
AF="COCO"
VF="images/blurred"
python semDINO.py --exp_dir $MROOT/$MNAME --root $ROOT --Sname $SNAME --afiles $AF --vfiles $VF

