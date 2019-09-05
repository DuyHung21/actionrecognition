#!/bin/bash
echo $1 $2

echo "{" >> $2
flag=0
for folder in $1/*; do
  echo "In folder:\" ${folder}"
  for file in $folder/*; do
    num_frame=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of default=nokey=1:noprint_wrappers=1 $file)
    if [ $flag -eq 1 ]; then
      echo "," >> $2
    else
      let flag=1
    fi
    echo -n "\"${file}\": {\"frames\": ${num_frame}, \"class\": \"`basename ${folder}`\"}" >> $2
  done
done
echo "}" >> $2
