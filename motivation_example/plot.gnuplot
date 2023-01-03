set term postscript eps color size 7, 3 font ",30"
outFile="effect_tuning.eps"
set lmargin 10.0
set tmargin 0.0
set bmargin 0.0
set rmargin 0.0
set datafile separator ','
set rmargin 0
set output outFile
set xrange [0:6]
set yrange [-1.4:2.5]
set size ratio -1
set xtics font ",20"
set ytics font ",20"
unset xtics
unset ytics
resFolder="."
set border 0
obst_list=system("ls -1B ".resFolder."/obst_*")
set key left top outside
plot "res1.csv" using "fk2_x":"fk2_y" with lines lw 10 title "greedy fabric", \
 "res2.csv" using "fk2_x":"fk2_y" with lines lw 10 title "conservative fabric", \
  "initState.csv" using 1:2 with points lt 7 lc rgb "green" ps 3 title "start", \
  "goalState.csv" using 1:2 with points lt 7 lc rgb "blue" ps 3 title "goal", \
  "obst.csv" using 1:2 with points lt 7 lc rgb "black" ps 13 notitle

