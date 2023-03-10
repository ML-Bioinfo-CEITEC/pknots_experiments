#!/bin/bash

for GA in 4 8 # Eva 4, 8; Denisa 16, 32
do
	for WD in 0 0.01 0.05
	do
		for LR in 0.0001 0.00001 0.000001
		do
            echo "==============================="
            echo "GA $GA LR $LR WD $WD"
            echo "==============================="
			./M1_train.py -GA $GA -LR $LR -WD $WD
		done
	done
done
