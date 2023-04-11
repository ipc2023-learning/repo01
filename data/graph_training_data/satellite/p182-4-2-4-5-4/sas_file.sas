begin_version
3
end_version
begin_metric
0
end_metric
21
begin_variable
var0
-1
2
Atom power_avail(satellite3)
NegatedAtom power_avail(satellite3)
end_variable
begin_variable
var1
-1
2
Atom power_on(instrument5)
NegatedAtom power_on(instrument5)
end_variable
begin_variable
var2
-1
2
Atom power_avail(satellite2)
NegatedAtom power_avail(satellite2)
end_variable
begin_variable
var3
-1
2
Atom power_on(instrument4)
NegatedAtom power_on(instrument4)
end_variable
begin_variable
var4
-1
2
Atom power_on(instrument2)
NegatedAtom power_on(instrument2)
end_variable
begin_variable
var5
-1
2
Atom power_on(instrument3)
NegatedAtom power_on(instrument3)
end_variable
begin_variable
var6
-1
2
Atom power_avail(satellite1)
NegatedAtom power_avail(satellite1)
end_variable
begin_variable
var7
-1
2
Atom power_on(instrument0)
NegatedAtom power_on(instrument0)
end_variable
begin_variable
var8
-1
2
Atom power_on(instrument1)
NegatedAtom power_on(instrument1)
end_variable
begin_variable
var9
-1
2
Atom power_avail(satellite0)
NegatedAtom power_avail(satellite0)
end_variable
begin_variable
var10
-1
9
Atom pointing(satellite3, groundstation0)
Atom pointing(satellite3, groundstation1)
Atom pointing(satellite3, groundstation4)
Atom pointing(satellite3, phenomenon6)
Atom pointing(satellite3, phenomenon7)
Atom pointing(satellite3, planet8)
Atom pointing(satellite3, star2)
Atom pointing(satellite3, star3)
Atom pointing(satellite3, star5)
end_variable
begin_variable
var11
-1
9
Atom pointing(satellite2, groundstation0)
Atom pointing(satellite2, groundstation1)
Atom pointing(satellite2, groundstation4)
Atom pointing(satellite2, phenomenon6)
Atom pointing(satellite2, phenomenon7)
Atom pointing(satellite2, planet8)
Atom pointing(satellite2, star2)
Atom pointing(satellite2, star3)
Atom pointing(satellite2, star5)
end_variable
begin_variable
var12
-1
9
Atom pointing(satellite1, groundstation0)
Atom pointing(satellite1, groundstation1)
Atom pointing(satellite1, groundstation4)
Atom pointing(satellite1, phenomenon6)
Atom pointing(satellite1, phenomenon7)
Atom pointing(satellite1, planet8)
Atom pointing(satellite1, star2)
Atom pointing(satellite1, star3)
Atom pointing(satellite1, star5)
end_variable
begin_variable
var13
-1
9
Atom pointing(satellite0, groundstation0)
Atom pointing(satellite0, groundstation1)
Atom pointing(satellite0, groundstation4)
Atom pointing(satellite0, phenomenon6)
Atom pointing(satellite0, phenomenon7)
Atom pointing(satellite0, planet8)
Atom pointing(satellite0, star2)
Atom pointing(satellite0, star3)
Atom pointing(satellite0, star5)
end_variable
begin_variable
var14
-1
2
Atom calibrated(instrument5)
NegatedAtom calibrated(instrument5)
end_variable
begin_variable
var15
-1
2
Atom calibrated(instrument4)
NegatedAtom calibrated(instrument4)
end_variable
begin_variable
var16
-1
2
Atom calibrated(instrument3)
NegatedAtom calibrated(instrument3)
end_variable
begin_variable
var17
-1
2
Atom calibrated(instrument1)
NegatedAtom calibrated(instrument1)
end_variable
begin_variable
var18
-1
2
Atom have_image(star5, infrared2)
NegatedAtom have_image(star5, infrared2)
end_variable
begin_variable
var19
-1
2
Atom have_image(phenomenon7, infrared2)
NegatedAtom have_image(phenomenon7, infrared2)
end_variable
begin_variable
var20
-1
2
Atom have_image(phenomenon6, infrared2)
NegatedAtom have_image(phenomenon6, infrared2)
end_variable
0
begin_state
0
1
0
1
1
1
0
1
1
0
5
8
0
5
1
1
1
1
1
1
1
end_state
begin_goal
3
18 0
19 0
20 0
end_goal
316
begin_operator
calibrate satellite0 instrument1 star2
2
13 6
8 0
1
0 17 -1 0
1
end_operator
begin_operator
calibrate satellite1 instrument3 star2
2
12 6
5 0
1
0 16 -1 0
1
end_operator
begin_operator
calibrate satellite2 instrument4 groundstation4
2
11 2
3 0
1
0 15 -1 0
1
end_operator
begin_operator
calibrate satellite3 instrument5 groundstation4
2
10 2
1 0
1
0 14 -1 0
1
end_operator
begin_operator
switch_off instrument0 satellite0
0
2
0 9 -1 0
0 7 0 1
1
end_operator
begin_operator
switch_off instrument1 satellite0
0
2
0 9 -1 0
0 8 0 1
1
end_operator
begin_operator
switch_off instrument2 satellite1
0
2
0 6 -1 0
0 4 0 1
1
end_operator
begin_operator
switch_off instrument3 satellite1
0
2
0 6 -1 0
0 5 0 1
1
end_operator
begin_operator
switch_off instrument4 satellite2
0
2
0 2 -1 0
0 3 0 1
1
end_operator
begin_operator
switch_off instrument5 satellite3
0
2
0 0 -1 0
0 1 0 1
1
end_operator
begin_operator
switch_on instrument0 satellite0
0
2
0 9 0 1
0 7 -1 0
1
end_operator
begin_operator
switch_on instrument1 satellite0
0
3
0 17 -1 1
0 9 0 1
0 8 -1 0
1
end_operator
begin_operator
switch_on instrument2 satellite1
0
2
0 6 0 1
0 4 -1 0
1
end_operator
begin_operator
switch_on instrument3 satellite1
0
3
0 16 -1 1
0 6 0 1
0 5 -1 0
1
end_operator
begin_operator
switch_on instrument4 satellite2
0
3
0 15 -1 1
0 2 0 1
0 3 -1 0
1
end_operator
begin_operator
switch_on instrument5 satellite3
0
3
0 14 -1 1
0 0 0 1
0 1 -1 0
1
end_operator
begin_operator
take_image satellite0 phenomenon6 instrument1 infrared2
3
17 0
13 3
8 0
1
0 20 -1 0
1
end_operator
begin_operator
take_image satellite0 phenomenon7 instrument1 infrared2
3
17 0
13 4
8 0
1
0 19 -1 0
1
end_operator
begin_operator
take_image satellite0 star5 instrument1 infrared2
3
17 0
13 8
8 0
1
0 18 -1 0
1
end_operator
begin_operator
take_image satellite1 phenomenon6 instrument3 infrared2
3
16 0
12 3
5 0
1
0 20 -1 0
1
end_operator
begin_operator
take_image satellite1 phenomenon7 instrument3 infrared2
3
16 0
12 4
5 0
1
0 19 -1 0
1
end_operator
begin_operator
take_image satellite1 star5 instrument3 infrared2
3
16 0
12 8
5 0
1
0 18 -1 0
1
end_operator
begin_operator
take_image satellite2 phenomenon6 instrument4 infrared2
3
15 0
11 3
3 0
1
0 20 -1 0
1
end_operator
begin_operator
take_image satellite2 phenomenon7 instrument4 infrared2
3
15 0
11 4
3 0
1
0 19 -1 0
1
end_operator
begin_operator
take_image satellite2 star5 instrument4 infrared2
3
15 0
11 8
3 0
1
0 18 -1 0
1
end_operator
begin_operator
take_image satellite3 phenomenon6 instrument5 infrared2
3
14 0
10 3
1 0
1
0 20 -1 0
1
end_operator
begin_operator
take_image satellite3 phenomenon7 instrument5 infrared2
3
14 0
10 4
1 0
1
0 19 -1 0
1
end_operator
begin_operator
take_image satellite3 star5 instrument5 infrared2
3
14 0
10 8
1 0
1
0 18 -1 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 groundstation1
0
1
0 13 1 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 groundstation4
0
1
0 13 2 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 phenomenon6
0
1
0 13 3 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 phenomenon7
0
1
0 13 4 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 planet8
0
1
0 13 5 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 star2
0
1
0 13 6 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 star3
0
1
0 13 7 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 star5
0
1
0 13 8 0
1
end_operator
begin_operator
turn_to satellite0 groundstation1 groundstation0
0
1
0 13 0 1
1
end_operator
begin_operator
turn_to satellite0 groundstation1 groundstation4
0
1
0 13 2 1
1
end_operator
begin_operator
turn_to satellite0 groundstation1 phenomenon6
0
1
0 13 3 1
1
end_operator
begin_operator
turn_to satellite0 groundstation1 phenomenon7
0
1
0 13 4 1
1
end_operator
begin_operator
turn_to satellite0 groundstation1 planet8
0
1
0 13 5 1
1
end_operator
begin_operator
turn_to satellite0 groundstation1 star2
0
1
0 13 6 1
1
end_operator
begin_operator
turn_to satellite0 groundstation1 star3
0
1
0 13 7 1
1
end_operator
begin_operator
turn_to satellite0 groundstation1 star5
0
1
0 13 8 1
1
end_operator
begin_operator
turn_to satellite0 groundstation4 groundstation0
0
1
0 13 0 2
1
end_operator
begin_operator
turn_to satellite0 groundstation4 groundstation1
0
1
0 13 1 2
1
end_operator
begin_operator
turn_to satellite0 groundstation4 phenomenon6
0
1
0 13 3 2
1
end_operator
begin_operator
turn_to satellite0 groundstation4 phenomenon7
0
1
0 13 4 2
1
end_operator
begin_operator
turn_to satellite0 groundstation4 planet8
0
1
0 13 5 2
1
end_operator
begin_operator
turn_to satellite0 groundstation4 star2
0
1
0 13 6 2
1
end_operator
begin_operator
turn_to satellite0 groundstation4 star3
0
1
0 13 7 2
1
end_operator
begin_operator
turn_to satellite0 groundstation4 star5
0
1
0 13 8 2
1
end_operator
begin_operator
turn_to satellite0 phenomenon6 groundstation0
0
1
0 13 0 3
1
end_operator
begin_operator
turn_to satellite0 phenomenon6 groundstation1
0
1
0 13 1 3
1
end_operator
begin_operator
turn_to satellite0 phenomenon6 groundstation4
0
1
0 13 2 3
1
end_operator
begin_operator
turn_to satellite0 phenomenon6 phenomenon7
0
1
0 13 4 3
1
end_operator
begin_operator
turn_to satellite0 phenomenon6 planet8
0
1
0 13 5 3
1
end_operator
begin_operator
turn_to satellite0 phenomenon6 star2
0
1
0 13 6 3
1
end_operator
begin_operator
turn_to satellite0 phenomenon6 star3
0
1
0 13 7 3
1
end_operator
begin_operator
turn_to satellite0 phenomenon6 star5
0
1
0 13 8 3
1
end_operator
begin_operator
turn_to satellite0 phenomenon7 groundstation0
0
1
0 13 0 4
1
end_operator
begin_operator
turn_to satellite0 phenomenon7 groundstation1
0
1
0 13 1 4
1
end_operator
begin_operator
turn_to satellite0 phenomenon7 groundstation4
0
1
0 13 2 4
1
end_operator
begin_operator
turn_to satellite0 phenomenon7 phenomenon6
0
1
0 13 3 4
1
end_operator
begin_operator
turn_to satellite0 phenomenon7 planet8
0
1
0 13 5 4
1
end_operator
begin_operator
turn_to satellite0 phenomenon7 star2
0
1
0 13 6 4
1
end_operator
begin_operator
turn_to satellite0 phenomenon7 star3
0
1
0 13 7 4
1
end_operator
begin_operator
turn_to satellite0 phenomenon7 star5
0
1
0 13 8 4
1
end_operator
begin_operator
turn_to satellite0 planet8 groundstation0
0
1
0 13 0 5
1
end_operator
begin_operator
turn_to satellite0 planet8 groundstation1
0
1
0 13 1 5
1
end_operator
begin_operator
turn_to satellite0 planet8 groundstation4
0
1
0 13 2 5
1
end_operator
begin_operator
turn_to satellite0 planet8 phenomenon6
0
1
0 13 3 5
1
end_operator
begin_operator
turn_to satellite0 planet8 phenomenon7
0
1
0 13 4 5
1
end_operator
begin_operator
turn_to satellite0 planet8 star2
0
1
0 13 6 5
1
end_operator
begin_operator
turn_to satellite0 planet8 star3
0
1
0 13 7 5
1
end_operator
begin_operator
turn_to satellite0 planet8 star5
0
1
0 13 8 5
1
end_operator
begin_operator
turn_to satellite0 star2 groundstation0
0
1
0 13 0 6
1
end_operator
begin_operator
turn_to satellite0 star2 groundstation1
0
1
0 13 1 6
1
end_operator
begin_operator
turn_to satellite0 star2 groundstation4
0
1
0 13 2 6
1
end_operator
begin_operator
turn_to satellite0 star2 phenomenon6
0
1
0 13 3 6
1
end_operator
begin_operator
turn_to satellite0 star2 phenomenon7
0
1
0 13 4 6
1
end_operator
begin_operator
turn_to satellite0 star2 planet8
0
1
0 13 5 6
1
end_operator
begin_operator
turn_to satellite0 star2 star3
0
1
0 13 7 6
1
end_operator
begin_operator
turn_to satellite0 star2 star5
0
1
0 13 8 6
1
end_operator
begin_operator
turn_to satellite0 star3 groundstation0
0
1
0 13 0 7
1
end_operator
begin_operator
turn_to satellite0 star3 groundstation1
0
1
0 13 1 7
1
end_operator
begin_operator
turn_to satellite0 star3 groundstation4
0
1
0 13 2 7
1
end_operator
begin_operator
turn_to satellite0 star3 phenomenon6
0
1
0 13 3 7
1
end_operator
begin_operator
turn_to satellite0 star3 phenomenon7
0
1
0 13 4 7
1
end_operator
begin_operator
turn_to satellite0 star3 planet8
0
1
0 13 5 7
1
end_operator
begin_operator
turn_to satellite0 star3 star2
0
1
0 13 6 7
1
end_operator
begin_operator
turn_to satellite0 star3 star5
0
1
0 13 8 7
1
end_operator
begin_operator
turn_to satellite0 star5 groundstation0
0
1
0 13 0 8
1
end_operator
begin_operator
turn_to satellite0 star5 groundstation1
0
1
0 13 1 8
1
end_operator
begin_operator
turn_to satellite0 star5 groundstation4
0
1
0 13 2 8
1
end_operator
begin_operator
turn_to satellite0 star5 phenomenon6
0
1
0 13 3 8
1
end_operator
begin_operator
turn_to satellite0 star5 phenomenon7
0
1
0 13 4 8
1
end_operator
begin_operator
turn_to satellite0 star5 planet8
0
1
0 13 5 8
1
end_operator
begin_operator
turn_to satellite0 star5 star2
0
1
0 13 6 8
1
end_operator
begin_operator
turn_to satellite0 star5 star3
0
1
0 13 7 8
1
end_operator
begin_operator
turn_to satellite1 groundstation0 groundstation1
0
1
0 12 1 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 groundstation4
0
1
0 12 2 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 phenomenon6
0
1
0 12 3 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 phenomenon7
0
1
0 12 4 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 planet8
0
1
0 12 5 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 star2
0
1
0 12 6 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 star3
0
1
0 12 7 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 star5
0
1
0 12 8 0
1
end_operator
begin_operator
turn_to satellite1 groundstation1 groundstation0
0
1
0 12 0 1
1
end_operator
begin_operator
turn_to satellite1 groundstation1 groundstation4
0
1
0 12 2 1
1
end_operator
begin_operator
turn_to satellite1 groundstation1 phenomenon6
0
1
0 12 3 1
1
end_operator
begin_operator
turn_to satellite1 groundstation1 phenomenon7
0
1
0 12 4 1
1
end_operator
begin_operator
turn_to satellite1 groundstation1 planet8
0
1
0 12 5 1
1
end_operator
begin_operator
turn_to satellite1 groundstation1 star2
0
1
0 12 6 1
1
end_operator
begin_operator
turn_to satellite1 groundstation1 star3
0
1
0 12 7 1
1
end_operator
begin_operator
turn_to satellite1 groundstation1 star5
0
1
0 12 8 1
1
end_operator
begin_operator
turn_to satellite1 groundstation4 groundstation0
0
1
0 12 0 2
1
end_operator
begin_operator
turn_to satellite1 groundstation4 groundstation1
0
1
0 12 1 2
1
end_operator
begin_operator
turn_to satellite1 groundstation4 phenomenon6
0
1
0 12 3 2
1
end_operator
begin_operator
turn_to satellite1 groundstation4 phenomenon7
0
1
0 12 4 2
1
end_operator
begin_operator
turn_to satellite1 groundstation4 planet8
0
1
0 12 5 2
1
end_operator
begin_operator
turn_to satellite1 groundstation4 star2
0
1
0 12 6 2
1
end_operator
begin_operator
turn_to satellite1 groundstation4 star3
0
1
0 12 7 2
1
end_operator
begin_operator
turn_to satellite1 groundstation4 star5
0
1
0 12 8 2
1
end_operator
begin_operator
turn_to satellite1 phenomenon6 groundstation0
0
1
0 12 0 3
1
end_operator
begin_operator
turn_to satellite1 phenomenon6 groundstation1
0
1
0 12 1 3
1
end_operator
begin_operator
turn_to satellite1 phenomenon6 groundstation4
0
1
0 12 2 3
1
end_operator
begin_operator
turn_to satellite1 phenomenon6 phenomenon7
0
1
0 12 4 3
1
end_operator
begin_operator
turn_to satellite1 phenomenon6 planet8
0
1
0 12 5 3
1
end_operator
begin_operator
turn_to satellite1 phenomenon6 star2
0
1
0 12 6 3
1
end_operator
begin_operator
turn_to satellite1 phenomenon6 star3
0
1
0 12 7 3
1
end_operator
begin_operator
turn_to satellite1 phenomenon6 star5
0
1
0 12 8 3
1
end_operator
begin_operator
turn_to satellite1 phenomenon7 groundstation0
0
1
0 12 0 4
1
end_operator
begin_operator
turn_to satellite1 phenomenon7 groundstation1
0
1
0 12 1 4
1
end_operator
begin_operator
turn_to satellite1 phenomenon7 groundstation4
0
1
0 12 2 4
1
end_operator
begin_operator
turn_to satellite1 phenomenon7 phenomenon6
0
1
0 12 3 4
1
end_operator
begin_operator
turn_to satellite1 phenomenon7 planet8
0
1
0 12 5 4
1
end_operator
begin_operator
turn_to satellite1 phenomenon7 star2
0
1
0 12 6 4
1
end_operator
begin_operator
turn_to satellite1 phenomenon7 star3
0
1
0 12 7 4
1
end_operator
begin_operator
turn_to satellite1 phenomenon7 star5
0
1
0 12 8 4
1
end_operator
begin_operator
turn_to satellite1 planet8 groundstation0
0
1
0 12 0 5
1
end_operator
begin_operator
turn_to satellite1 planet8 groundstation1
0
1
0 12 1 5
1
end_operator
begin_operator
turn_to satellite1 planet8 groundstation4
0
1
0 12 2 5
1
end_operator
begin_operator
turn_to satellite1 planet8 phenomenon6
0
1
0 12 3 5
1
end_operator
begin_operator
turn_to satellite1 planet8 phenomenon7
0
1
0 12 4 5
1
end_operator
begin_operator
turn_to satellite1 planet8 star2
0
1
0 12 6 5
1
end_operator
begin_operator
turn_to satellite1 planet8 star3
0
1
0 12 7 5
1
end_operator
begin_operator
turn_to satellite1 planet8 star5
0
1
0 12 8 5
1
end_operator
begin_operator
turn_to satellite1 star2 groundstation0
0
1
0 12 0 6
1
end_operator
begin_operator
turn_to satellite1 star2 groundstation1
0
1
0 12 1 6
1
end_operator
begin_operator
turn_to satellite1 star2 groundstation4
0
1
0 12 2 6
1
end_operator
begin_operator
turn_to satellite1 star2 phenomenon6
0
1
0 12 3 6
1
end_operator
begin_operator
turn_to satellite1 star2 phenomenon7
0
1
0 12 4 6
1
end_operator
begin_operator
turn_to satellite1 star2 planet8
0
1
0 12 5 6
1
end_operator
begin_operator
turn_to satellite1 star2 star3
0
1
0 12 7 6
1
end_operator
begin_operator
turn_to satellite1 star2 star5
0
1
0 12 8 6
1
end_operator
begin_operator
turn_to satellite1 star3 groundstation0
0
1
0 12 0 7
1
end_operator
begin_operator
turn_to satellite1 star3 groundstation1
0
1
0 12 1 7
1
end_operator
begin_operator
turn_to satellite1 star3 groundstation4
0
1
0 12 2 7
1
end_operator
begin_operator
turn_to satellite1 star3 phenomenon6
0
1
0 12 3 7
1
end_operator
begin_operator
turn_to satellite1 star3 phenomenon7
0
1
0 12 4 7
1
end_operator
begin_operator
turn_to satellite1 star3 planet8
0
1
0 12 5 7
1
end_operator
begin_operator
turn_to satellite1 star3 star2
0
1
0 12 6 7
1
end_operator
begin_operator
turn_to satellite1 star3 star5
0
1
0 12 8 7
1
end_operator
begin_operator
turn_to satellite1 star5 groundstation0
0
1
0 12 0 8
1
end_operator
begin_operator
turn_to satellite1 star5 groundstation1
0
1
0 12 1 8
1
end_operator
begin_operator
turn_to satellite1 star5 groundstation4
0
1
0 12 2 8
1
end_operator
begin_operator
turn_to satellite1 star5 phenomenon6
0
1
0 12 3 8
1
end_operator
begin_operator
turn_to satellite1 star5 phenomenon7
0
1
0 12 4 8
1
end_operator
begin_operator
turn_to satellite1 star5 planet8
0
1
0 12 5 8
1
end_operator
begin_operator
turn_to satellite1 star5 star2
0
1
0 12 6 8
1
end_operator
begin_operator
turn_to satellite1 star5 star3
0
1
0 12 7 8
1
end_operator
begin_operator
turn_to satellite2 groundstation0 groundstation1
0
1
0 11 1 0
1
end_operator
begin_operator
turn_to satellite2 groundstation0 groundstation4
0
1
0 11 2 0
1
end_operator
begin_operator
turn_to satellite2 groundstation0 phenomenon6
0
1
0 11 3 0
1
end_operator
begin_operator
turn_to satellite2 groundstation0 phenomenon7
0
1
0 11 4 0
1
end_operator
begin_operator
turn_to satellite2 groundstation0 planet8
0
1
0 11 5 0
1
end_operator
begin_operator
turn_to satellite2 groundstation0 star2
0
1
0 11 6 0
1
end_operator
begin_operator
turn_to satellite2 groundstation0 star3
0
1
0 11 7 0
1
end_operator
begin_operator
turn_to satellite2 groundstation0 star5
0
1
0 11 8 0
1
end_operator
begin_operator
turn_to satellite2 groundstation1 groundstation0
0
1
0 11 0 1
1
end_operator
begin_operator
turn_to satellite2 groundstation1 groundstation4
0
1
0 11 2 1
1
end_operator
begin_operator
turn_to satellite2 groundstation1 phenomenon6
0
1
0 11 3 1
1
end_operator
begin_operator
turn_to satellite2 groundstation1 phenomenon7
0
1
0 11 4 1
1
end_operator
begin_operator
turn_to satellite2 groundstation1 planet8
0
1
0 11 5 1
1
end_operator
begin_operator
turn_to satellite2 groundstation1 star2
0
1
0 11 6 1
1
end_operator
begin_operator
turn_to satellite2 groundstation1 star3
0
1
0 11 7 1
1
end_operator
begin_operator
turn_to satellite2 groundstation1 star5
0
1
0 11 8 1
1
end_operator
begin_operator
turn_to satellite2 groundstation4 groundstation0
0
1
0 11 0 2
1
end_operator
begin_operator
turn_to satellite2 groundstation4 groundstation1
0
1
0 11 1 2
1
end_operator
begin_operator
turn_to satellite2 groundstation4 phenomenon6
0
1
0 11 3 2
1
end_operator
begin_operator
turn_to satellite2 groundstation4 phenomenon7
0
1
0 11 4 2
1
end_operator
begin_operator
turn_to satellite2 groundstation4 planet8
0
1
0 11 5 2
1
end_operator
begin_operator
turn_to satellite2 groundstation4 star2
0
1
0 11 6 2
1
end_operator
begin_operator
turn_to satellite2 groundstation4 star3
0
1
0 11 7 2
1
end_operator
begin_operator
turn_to satellite2 groundstation4 star5
0
1
0 11 8 2
1
end_operator
begin_operator
turn_to satellite2 phenomenon6 groundstation0
0
1
0 11 0 3
1
end_operator
begin_operator
turn_to satellite2 phenomenon6 groundstation1
0
1
0 11 1 3
1
end_operator
begin_operator
turn_to satellite2 phenomenon6 groundstation4
0
1
0 11 2 3
1
end_operator
begin_operator
turn_to satellite2 phenomenon6 phenomenon7
0
1
0 11 4 3
1
end_operator
begin_operator
turn_to satellite2 phenomenon6 planet8
0
1
0 11 5 3
1
end_operator
begin_operator
turn_to satellite2 phenomenon6 star2
0
1
0 11 6 3
1
end_operator
begin_operator
turn_to satellite2 phenomenon6 star3
0
1
0 11 7 3
1
end_operator
begin_operator
turn_to satellite2 phenomenon6 star5
0
1
0 11 8 3
1
end_operator
begin_operator
turn_to satellite2 phenomenon7 groundstation0
0
1
0 11 0 4
1
end_operator
begin_operator
turn_to satellite2 phenomenon7 groundstation1
0
1
0 11 1 4
1
end_operator
begin_operator
turn_to satellite2 phenomenon7 groundstation4
0
1
0 11 2 4
1
end_operator
begin_operator
turn_to satellite2 phenomenon7 phenomenon6
0
1
0 11 3 4
1
end_operator
begin_operator
turn_to satellite2 phenomenon7 planet8
0
1
0 11 5 4
1
end_operator
begin_operator
turn_to satellite2 phenomenon7 star2
0
1
0 11 6 4
1
end_operator
begin_operator
turn_to satellite2 phenomenon7 star3
0
1
0 11 7 4
1
end_operator
begin_operator
turn_to satellite2 phenomenon7 star5
0
1
0 11 8 4
1
end_operator
begin_operator
turn_to satellite2 planet8 groundstation0
0
1
0 11 0 5
1
end_operator
begin_operator
turn_to satellite2 planet8 groundstation1
0
1
0 11 1 5
1
end_operator
begin_operator
turn_to satellite2 planet8 groundstation4
0
1
0 11 2 5
1
end_operator
begin_operator
turn_to satellite2 planet8 phenomenon6
0
1
0 11 3 5
1
end_operator
begin_operator
turn_to satellite2 planet8 phenomenon7
0
1
0 11 4 5
1
end_operator
begin_operator
turn_to satellite2 planet8 star2
0
1
0 11 6 5
1
end_operator
begin_operator
turn_to satellite2 planet8 star3
0
1
0 11 7 5
1
end_operator
begin_operator
turn_to satellite2 planet8 star5
0
1
0 11 8 5
1
end_operator
begin_operator
turn_to satellite2 star2 groundstation0
0
1
0 11 0 6
1
end_operator
begin_operator
turn_to satellite2 star2 groundstation1
0
1
0 11 1 6
1
end_operator
begin_operator
turn_to satellite2 star2 groundstation4
0
1
0 11 2 6
1
end_operator
begin_operator
turn_to satellite2 star2 phenomenon6
0
1
0 11 3 6
1
end_operator
begin_operator
turn_to satellite2 star2 phenomenon7
0
1
0 11 4 6
1
end_operator
begin_operator
turn_to satellite2 star2 planet8
0
1
0 11 5 6
1
end_operator
begin_operator
turn_to satellite2 star2 star3
0
1
0 11 7 6
1
end_operator
begin_operator
turn_to satellite2 star2 star5
0
1
0 11 8 6
1
end_operator
begin_operator
turn_to satellite2 star3 groundstation0
0
1
0 11 0 7
1
end_operator
begin_operator
turn_to satellite2 star3 groundstation1
0
1
0 11 1 7
1
end_operator
begin_operator
turn_to satellite2 star3 groundstation4
0
1
0 11 2 7
1
end_operator
begin_operator
turn_to satellite2 star3 phenomenon6
0
1
0 11 3 7
1
end_operator
begin_operator
turn_to satellite2 star3 phenomenon7
0
1
0 11 4 7
1
end_operator
begin_operator
turn_to satellite2 star3 planet8
0
1
0 11 5 7
1
end_operator
begin_operator
turn_to satellite2 star3 star2
0
1
0 11 6 7
1
end_operator
begin_operator
turn_to satellite2 star3 star5
0
1
0 11 8 7
1
end_operator
begin_operator
turn_to satellite2 star5 groundstation0
0
1
0 11 0 8
1
end_operator
begin_operator
turn_to satellite2 star5 groundstation1
0
1
0 11 1 8
1
end_operator
begin_operator
turn_to satellite2 star5 groundstation4
0
1
0 11 2 8
1
end_operator
begin_operator
turn_to satellite2 star5 phenomenon6
0
1
0 11 3 8
1
end_operator
begin_operator
turn_to satellite2 star5 phenomenon7
0
1
0 11 4 8
1
end_operator
begin_operator
turn_to satellite2 star5 planet8
0
1
0 11 5 8
1
end_operator
begin_operator
turn_to satellite2 star5 star2
0
1
0 11 6 8
1
end_operator
begin_operator
turn_to satellite2 star5 star3
0
1
0 11 7 8
1
end_operator
begin_operator
turn_to satellite3 groundstation0 groundstation1
0
1
0 10 1 0
1
end_operator
begin_operator
turn_to satellite3 groundstation0 groundstation4
0
1
0 10 2 0
1
end_operator
begin_operator
turn_to satellite3 groundstation0 phenomenon6
0
1
0 10 3 0
1
end_operator
begin_operator
turn_to satellite3 groundstation0 phenomenon7
0
1
0 10 4 0
1
end_operator
begin_operator
turn_to satellite3 groundstation0 planet8
0
1
0 10 5 0
1
end_operator
begin_operator
turn_to satellite3 groundstation0 star2
0
1
0 10 6 0
1
end_operator
begin_operator
turn_to satellite3 groundstation0 star3
0
1
0 10 7 0
1
end_operator
begin_operator
turn_to satellite3 groundstation0 star5
0
1
0 10 8 0
1
end_operator
begin_operator
turn_to satellite3 groundstation1 groundstation0
0
1
0 10 0 1
1
end_operator
begin_operator
turn_to satellite3 groundstation1 groundstation4
0
1
0 10 2 1
1
end_operator
begin_operator
turn_to satellite3 groundstation1 phenomenon6
0
1
0 10 3 1
1
end_operator
begin_operator
turn_to satellite3 groundstation1 phenomenon7
0
1
0 10 4 1
1
end_operator
begin_operator
turn_to satellite3 groundstation1 planet8
0
1
0 10 5 1
1
end_operator
begin_operator
turn_to satellite3 groundstation1 star2
0
1
0 10 6 1
1
end_operator
begin_operator
turn_to satellite3 groundstation1 star3
0
1
0 10 7 1
1
end_operator
begin_operator
turn_to satellite3 groundstation1 star5
0
1
0 10 8 1
1
end_operator
begin_operator
turn_to satellite3 groundstation4 groundstation0
0
1
0 10 0 2
1
end_operator
begin_operator
turn_to satellite3 groundstation4 groundstation1
0
1
0 10 1 2
1
end_operator
begin_operator
turn_to satellite3 groundstation4 phenomenon6
0
1
0 10 3 2
1
end_operator
begin_operator
turn_to satellite3 groundstation4 phenomenon7
0
1
0 10 4 2
1
end_operator
begin_operator
turn_to satellite3 groundstation4 planet8
0
1
0 10 5 2
1
end_operator
begin_operator
turn_to satellite3 groundstation4 star2
0
1
0 10 6 2
1
end_operator
begin_operator
turn_to satellite3 groundstation4 star3
0
1
0 10 7 2
1
end_operator
begin_operator
turn_to satellite3 groundstation4 star5
0
1
0 10 8 2
1
end_operator
begin_operator
turn_to satellite3 phenomenon6 groundstation0
0
1
0 10 0 3
1
end_operator
begin_operator
turn_to satellite3 phenomenon6 groundstation1
0
1
0 10 1 3
1
end_operator
begin_operator
turn_to satellite3 phenomenon6 groundstation4
0
1
0 10 2 3
1
end_operator
begin_operator
turn_to satellite3 phenomenon6 phenomenon7
0
1
0 10 4 3
1
end_operator
begin_operator
turn_to satellite3 phenomenon6 planet8
0
1
0 10 5 3
1
end_operator
begin_operator
turn_to satellite3 phenomenon6 star2
0
1
0 10 6 3
1
end_operator
begin_operator
turn_to satellite3 phenomenon6 star3
0
1
0 10 7 3
1
end_operator
begin_operator
turn_to satellite3 phenomenon6 star5
0
1
0 10 8 3
1
end_operator
begin_operator
turn_to satellite3 phenomenon7 groundstation0
0
1
0 10 0 4
1
end_operator
begin_operator
turn_to satellite3 phenomenon7 groundstation1
0
1
0 10 1 4
1
end_operator
begin_operator
turn_to satellite3 phenomenon7 groundstation4
0
1
0 10 2 4
1
end_operator
begin_operator
turn_to satellite3 phenomenon7 phenomenon6
0
1
0 10 3 4
1
end_operator
begin_operator
turn_to satellite3 phenomenon7 planet8
0
1
0 10 5 4
1
end_operator
begin_operator
turn_to satellite3 phenomenon7 star2
0
1
0 10 6 4
1
end_operator
begin_operator
turn_to satellite3 phenomenon7 star3
0
1
0 10 7 4
1
end_operator
begin_operator
turn_to satellite3 phenomenon7 star5
0
1
0 10 8 4
1
end_operator
begin_operator
turn_to satellite3 planet8 groundstation0
0
1
0 10 0 5
1
end_operator
begin_operator
turn_to satellite3 planet8 groundstation1
0
1
0 10 1 5
1
end_operator
begin_operator
turn_to satellite3 planet8 groundstation4
0
1
0 10 2 5
1
end_operator
begin_operator
turn_to satellite3 planet8 phenomenon6
0
1
0 10 3 5
1
end_operator
begin_operator
turn_to satellite3 planet8 phenomenon7
0
1
0 10 4 5
1
end_operator
begin_operator
turn_to satellite3 planet8 star2
0
1
0 10 6 5
1
end_operator
begin_operator
turn_to satellite3 planet8 star3
0
1
0 10 7 5
1
end_operator
begin_operator
turn_to satellite3 planet8 star5
0
1
0 10 8 5
1
end_operator
begin_operator
turn_to satellite3 star2 groundstation0
0
1
0 10 0 6
1
end_operator
begin_operator
turn_to satellite3 star2 groundstation1
0
1
0 10 1 6
1
end_operator
begin_operator
turn_to satellite3 star2 groundstation4
0
1
0 10 2 6
1
end_operator
begin_operator
turn_to satellite3 star2 phenomenon6
0
1
0 10 3 6
1
end_operator
begin_operator
turn_to satellite3 star2 phenomenon7
0
1
0 10 4 6
1
end_operator
begin_operator
turn_to satellite3 star2 planet8
0
1
0 10 5 6
1
end_operator
begin_operator
turn_to satellite3 star2 star3
0
1
0 10 7 6
1
end_operator
begin_operator
turn_to satellite3 star2 star5
0
1
0 10 8 6
1
end_operator
begin_operator
turn_to satellite3 star3 groundstation0
0
1
0 10 0 7
1
end_operator
begin_operator
turn_to satellite3 star3 groundstation1
0
1
0 10 1 7
1
end_operator
begin_operator
turn_to satellite3 star3 groundstation4
0
1
0 10 2 7
1
end_operator
begin_operator
turn_to satellite3 star3 phenomenon6
0
1
0 10 3 7
1
end_operator
begin_operator
turn_to satellite3 star3 phenomenon7
0
1
0 10 4 7
1
end_operator
begin_operator
turn_to satellite3 star3 planet8
0
1
0 10 5 7
1
end_operator
begin_operator
turn_to satellite3 star3 star2
0
1
0 10 6 7
1
end_operator
begin_operator
turn_to satellite3 star3 star5
0
1
0 10 8 7
1
end_operator
begin_operator
turn_to satellite3 star5 groundstation0
0
1
0 10 0 8
1
end_operator
begin_operator
turn_to satellite3 star5 groundstation1
0
1
0 10 1 8
1
end_operator
begin_operator
turn_to satellite3 star5 groundstation4
0
1
0 10 2 8
1
end_operator
begin_operator
turn_to satellite3 star5 phenomenon6
0
1
0 10 3 8
1
end_operator
begin_operator
turn_to satellite3 star5 phenomenon7
0
1
0 10 4 8
1
end_operator
begin_operator
turn_to satellite3 star5 planet8
0
1
0 10 5 8
1
end_operator
begin_operator
turn_to satellite3 star5 star2
0
1
0 10 6 8
1
end_operator
begin_operator
turn_to satellite3 star5 star3
0
1
0 10 7 8
1
end_operator
0
