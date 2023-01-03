set terminal postscript eps color size 2.0,2.5 font "RomanSerif.ttf" 20
set lmargin 5.5
set bmargin 2.5
set rmargin 1.0
set tmargin 0.5
set output "study_history.eps"
set datafile separator ","
set xlabel "#trials" offset 0,0.5
set yrang [0.2:0.8]
set ytics 0,0.2,1
set ylabel "objective" offset 1.8,0
set key center top
plot "study_values.csv" u 1:2 w lp lw 3 pt 7 title "trial value" ,\
  "study_values.csv" u 1:3 w l lw 5 title "best value" ,\
