#!/bin/bash

/data/code/pubu/src/scripts/comment_dump.py $*
/data/code/miner/src/genor/sumy_genor.py 5 /tmp/comments.$1.txt
