###
### helper script
###

if [ $# != 1 ]; then
  echo 'usage: ./process.sh [log.json]'
  exit
fi

INPUT=$1
OUTPUT=${INPUT%%.*}
OUTPUT=${OUTPUT}.xlsx

python process.py --input $INPUT --output $OUTPUT
