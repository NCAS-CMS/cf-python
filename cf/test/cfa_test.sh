# --------------------------------------------------------------------
# Test cfa
# --------------------------------------------------------------------
scripts=../../scripts
sample_files=../../docs/source/sample_files

test_file=delme_cfa.nc
test_dir=delme_cfa_dir

mkdir -p $test_dir

for opt in vs vm vc
do
#  echo $opt
  $scripts/cfa    -$opt $sample_files/*[np][cp] >/dev/null
  $scripts/cfa -1 -$opt $sample_files/*[np][cp] >/dev/null
  for f in $sample_files/*[np][cp]
  do
#    echo $f
    $scripts/cfa -$opt $f >/dev/null
    if [ $opt = vs ] ; then
      $scripts/cfa --overwrite -o $test_file $f
      $scripts/cfa --overwrite -d $test_dir  $f
    fi
  done
done

$scripts/cfa --overwrite -o $test_file $sample_files/*[np][cp]
$scripts/cfa --overwrite -d $test_dir  $sample_files/*[np][cp]

rm -fr $test_dir $test_file

