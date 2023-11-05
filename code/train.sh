for VARIABLE in $(seq 0 2)
do
    python unet.py --n $VARIABLE
done