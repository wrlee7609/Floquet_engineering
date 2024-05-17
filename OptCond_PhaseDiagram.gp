set pm3d interpolate 0,0
set view map
unset border
set tics out
set origin 0,0

set size ratio 1
set size 1,1
set palette defined (0 "black", 0.28 "black", 0.33 "blue", 0.42 "cyan", 0.5 "yellow", 0.55 "orange", 0.7 "red", 1.1 "pink", 1.2 "white" )
set title '' offset 0,-0.5
set xlabel ''
set ylabel ''
set xrange [0:1]
set yrange [0:1]
set zrange [0:1.2]
splot 'OptCond_Ec10_T002' u 1:2:($4/(1-0.5/10)) w p palette pt 5 ps 0 t ''