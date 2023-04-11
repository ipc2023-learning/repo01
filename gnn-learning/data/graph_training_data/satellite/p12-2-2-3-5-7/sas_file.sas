begin_version
3
end_version
begin_metric
0
end_metric
17
begin_variable
var0
-1
2
Atom power_on(instrument2)
NegatedAtom power_on(instrument2)
end_variable
begin_variable
var1
-1
2
Atom power_on(instrument3)
NegatedAtom power_on(instrument3)
end_variable
begin_variable
var2
-1
2
Atom power_avail(satellite1)
NegatedAtom power_avail(satellite1)
end_variable
begin_variable
var3
-1
2
Atom power_on(instrument0)
NegatedAtom power_on(instrument0)
end_variable
begin_variable
var4
-1
2
Atom power_on(instrument1)
NegatedAtom power_on(instrument1)
end_variable
begin_variable
var5
-1
2
Atom power_avail(satellite0)
NegatedAtom power_avail(satellite0)
end_variable
begin_variable
var6
-1
12
Atom pointing(satellite1, groundstation0)
Atom pointing(satellite1, groundstation3)
Atom pointing(satellite1, groundstation4)
Atom pointing(satellite1, phenomenon8)
Atom pointing(satellite1, planet10)
Atom pointing(satellite1, planet9)
Atom pointing(satellite1, star1)
Atom pointing(satellite1, star11)
Atom pointing(satellite1, star2)
Atom pointing(satellite1, star5)
Atom pointing(satellite1, star6)
Atom pointing(satellite1, star7)
end_variable
begin_variable
var7
-1
12
Atom pointing(satellite0, groundstation0)
Atom pointing(satellite0, groundstation3)
Atom pointing(satellite0, groundstation4)
Atom pointing(satellite0, phenomenon8)
Atom pointing(satellite0, planet10)
Atom pointing(satellite0, planet9)
Atom pointing(satellite0, star1)
Atom pointing(satellite0, star11)
Atom pointing(satellite0, star2)
Atom pointing(satellite0, star5)
Atom pointing(satellite0, star6)
Atom pointing(satellite0, star7)
end_variable
begin_variable
var8
-1
2
Atom calibrated(instrument3)
NegatedAtom calibrated(instrument3)
end_variable
begin_variable
var9
-1
2
Atom calibrated(instrument2)
NegatedAtom calibrated(instrument2)
end_variable
begin_variable
var10
-1
2
Atom calibrated(instrument1)
NegatedAtom calibrated(instrument1)
end_variable
begin_variable
var11
-1
2
Atom have_image(star7, thermograph0)
NegatedAtom have_image(star7, thermograph0)
end_variable
begin_variable
var12
-1
2
Atom calibrated(instrument0)
NegatedAtom calibrated(instrument0)
end_variable
begin_variable
var13
-1
2
Atom have_image(star6, image1)
NegatedAtom have_image(star6, image1)
end_variable
begin_variable
var14
-1
2
Atom have_image(star5, image1)
NegatedAtom have_image(star5, image1)
end_variable
begin_variable
var15
-1
2
Atom have_image(planet9, infrared2)
NegatedAtom have_image(planet9, infrared2)
end_variable
begin_variable
var16
-1
2
Atom have_image(planet10, infrared2)
NegatedAtom have_image(planet10, infrared2)
end_variable
0
begin_state
1
1
0
1
1
0
0
6
1
1
1
1
1
1
1
1
1
end_state
begin_goal
7
6 0
7 8
11 0
13 0
14 0
15 0
16 0
end_goal
290
begin_operator
calibrate satellite0 instrument0 groundstation4
2
7 2
3 0
1
0 12 -1 0
1
end_operator
begin_operator
calibrate satellite0 instrument1 groundstation3
2
7 1
4 0
1
0 10 -1 0
1
end_operator
begin_operator
calibrate satellite1 instrument2 groundstation4
2
6 2
0 0
1
0 9 -1 0
1
end_operator
begin_operator
calibrate satellite1 instrument3 groundstation4
2
6 2
1 0
1
0 8 -1 0
1
end_operator
begin_operator
switch_off instrument0 satellite0
0
2
0 5 -1 0
0 3 0 1
1
end_operator
begin_operator
switch_off instrument1 satellite0
0
2
0 5 -1 0
0 4 0 1
1
end_operator
begin_operator
switch_off instrument2 satellite1
0
2
0 2 -1 0
0 0 0 1
1
end_operator
begin_operator
switch_off instrument3 satellite1
0
2
0 2 -1 0
0 1 0 1
1
end_operator
begin_operator
switch_on instrument0 satellite0
0
3
0 12 -1 1
0 5 0 1
0 3 -1 0
1
end_operator
begin_operator
switch_on instrument1 satellite0
0
3
0 10 -1 1
0 5 0 1
0 4 -1 0
1
end_operator
begin_operator
switch_on instrument2 satellite1
0
3
0 9 -1 1
0 2 0 1
0 0 -1 0
1
end_operator
begin_operator
switch_on instrument3 satellite1
0
3
0 8 -1 1
0 2 0 1
0 1 -1 0
1
end_operator
begin_operator
take_image satellite0 planet10 instrument0 infrared2
3
12 0
7 4
3 0
1
0 16 -1 0
1
end_operator
begin_operator
take_image satellite0 planet10 instrument1 infrared2
3
10 0
7 4
4 0
1
0 16 -1 0
1
end_operator
begin_operator
take_image satellite0 planet9 instrument0 infrared2
3
12 0
7 5
3 0
1
0 15 -1 0
1
end_operator
begin_operator
take_image satellite0 planet9 instrument1 infrared2
3
10 0
7 5
4 0
1
0 15 -1 0
1
end_operator
begin_operator
take_image satellite0 star5 instrument0 image1
3
12 0
7 9
3 0
1
0 14 -1 0
1
end_operator
begin_operator
take_image satellite0 star5 instrument1 image1
3
10 0
7 9
4 0
1
0 14 -1 0
1
end_operator
begin_operator
take_image satellite0 star6 instrument0 image1
3
12 0
7 10
3 0
1
0 13 -1 0
1
end_operator
begin_operator
take_image satellite0 star6 instrument1 image1
3
10 0
7 10
4 0
1
0 13 -1 0
1
end_operator
begin_operator
take_image satellite0 star7 instrument1 thermograph0
3
10 0
7 11
4 0
1
0 11 -1 0
1
end_operator
begin_operator
take_image satellite1 star5 instrument2 image1
3
9 0
6 9
0 0
1
0 14 -1 0
1
end_operator
begin_operator
take_image satellite1 star5 instrument3 image1
3
8 0
6 9
1 0
1
0 14 -1 0
1
end_operator
begin_operator
take_image satellite1 star6 instrument2 image1
3
9 0
6 10
0 0
1
0 13 -1 0
1
end_operator
begin_operator
take_image satellite1 star6 instrument3 image1
3
8 0
6 10
1 0
1
0 13 -1 0
1
end_operator
begin_operator
take_image satellite1 star7 instrument3 thermograph0
3
8 0
6 11
1 0
1
0 11 -1 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 groundstation3
0
1
0 7 1 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 groundstation4
0
1
0 7 2 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 phenomenon8
0
1
0 7 3 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 planet10
0
1
0 7 4 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 planet9
0
1
0 7 5 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 star1
0
1
0 7 6 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 star11
0
1
0 7 7 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 star2
0
1
0 7 8 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 star5
0
1
0 7 9 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 star6
0
1
0 7 10 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 star7
0
1
0 7 11 0
1
end_operator
begin_operator
turn_to satellite0 groundstation3 groundstation0
0
1
0 7 0 1
1
end_operator
begin_operator
turn_to satellite0 groundstation3 groundstation4
0
1
0 7 2 1
1
end_operator
begin_operator
turn_to satellite0 groundstation3 phenomenon8
0
1
0 7 3 1
1
end_operator
begin_operator
turn_to satellite0 groundstation3 planet10
0
1
0 7 4 1
1
end_operator
begin_operator
turn_to satellite0 groundstation3 planet9
0
1
0 7 5 1
1
end_operator
begin_operator
turn_to satellite0 groundstation3 star1
0
1
0 7 6 1
1
end_operator
begin_operator
turn_to satellite0 groundstation3 star11
0
1
0 7 7 1
1
end_operator
begin_operator
turn_to satellite0 groundstation3 star2
0
1
0 7 8 1
1
end_operator
begin_operator
turn_to satellite0 groundstation3 star5
0
1
0 7 9 1
1
end_operator
begin_operator
turn_to satellite0 groundstation3 star6
0
1
0 7 10 1
1
end_operator
begin_operator
turn_to satellite0 groundstation3 star7
0
1
0 7 11 1
1
end_operator
begin_operator
turn_to satellite0 groundstation4 groundstation0
0
1
0 7 0 2
1
end_operator
begin_operator
turn_to satellite0 groundstation4 groundstation3
0
1
0 7 1 2
1
end_operator
begin_operator
turn_to satellite0 groundstation4 phenomenon8
0
1
0 7 3 2
1
end_operator
begin_operator
turn_to satellite0 groundstation4 planet10
0
1
0 7 4 2
1
end_operator
begin_operator
turn_to satellite0 groundstation4 planet9
0
1
0 7 5 2
1
end_operator
begin_operator
turn_to satellite0 groundstation4 star1
0
1
0 7 6 2
1
end_operator
begin_operator
turn_to satellite0 groundstation4 star11
0
1
0 7 7 2
1
end_operator
begin_operator
turn_to satellite0 groundstation4 star2
0
1
0 7 8 2
1
end_operator
begin_operator
turn_to satellite0 groundstation4 star5
0
1
0 7 9 2
1
end_operator
begin_operator
turn_to satellite0 groundstation4 star6
0
1
0 7 10 2
1
end_operator
begin_operator
turn_to satellite0 groundstation4 star7
0
1
0 7 11 2
1
end_operator
begin_operator
turn_to satellite0 phenomenon8 groundstation0
0
1
0 7 0 3
1
end_operator
begin_operator
turn_to satellite0 phenomenon8 groundstation3
0
1
0 7 1 3
1
end_operator
begin_operator
turn_to satellite0 phenomenon8 groundstation4
0
1
0 7 2 3
1
end_operator
begin_operator
turn_to satellite0 phenomenon8 planet10
0
1
0 7 4 3
1
end_operator
begin_operator
turn_to satellite0 phenomenon8 planet9
0
1
0 7 5 3
1
end_operator
begin_operator
turn_to satellite0 phenomenon8 star1
0
1
0 7 6 3
1
end_operator
begin_operator
turn_to satellite0 phenomenon8 star11
0
1
0 7 7 3
1
end_operator
begin_operator
turn_to satellite0 phenomenon8 star2
0
1
0 7 8 3
1
end_operator
begin_operator
turn_to satellite0 phenomenon8 star5
0
1
0 7 9 3
1
end_operator
begin_operator
turn_to satellite0 phenomenon8 star6
0
1
0 7 10 3
1
end_operator
begin_operator
turn_to satellite0 phenomenon8 star7
0
1
0 7 11 3
1
end_operator
begin_operator
turn_to satellite0 planet10 groundstation0
0
1
0 7 0 4
1
end_operator
begin_operator
turn_to satellite0 planet10 groundstation3
0
1
0 7 1 4
1
end_operator
begin_operator
turn_to satellite0 planet10 groundstation4
0
1
0 7 2 4
1
end_operator
begin_operator
turn_to satellite0 planet10 phenomenon8
0
1
0 7 3 4
1
end_operator
begin_operator
turn_to satellite0 planet10 planet9
0
1
0 7 5 4
1
end_operator
begin_operator
turn_to satellite0 planet10 star1
0
1
0 7 6 4
1
end_operator
begin_operator
turn_to satellite0 planet10 star11
0
1
0 7 7 4
1
end_operator
begin_operator
turn_to satellite0 planet10 star2
0
1
0 7 8 4
1
end_operator
begin_operator
turn_to satellite0 planet10 star5
0
1
0 7 9 4
1
end_operator
begin_operator
turn_to satellite0 planet10 star6
0
1
0 7 10 4
1
end_operator
begin_operator
turn_to satellite0 planet10 star7
0
1
0 7 11 4
1
end_operator
begin_operator
turn_to satellite0 planet9 groundstation0
0
1
0 7 0 5
1
end_operator
begin_operator
turn_to satellite0 planet9 groundstation3
0
1
0 7 1 5
1
end_operator
begin_operator
turn_to satellite0 planet9 groundstation4
0
1
0 7 2 5
1
end_operator
begin_operator
turn_to satellite0 planet9 phenomenon8
0
1
0 7 3 5
1
end_operator
begin_operator
turn_to satellite0 planet9 planet10
0
1
0 7 4 5
1
end_operator
begin_operator
turn_to satellite0 planet9 star1
0
1
0 7 6 5
1
end_operator
begin_operator
turn_to satellite0 planet9 star11
0
1
0 7 7 5
1
end_operator
begin_operator
turn_to satellite0 planet9 star2
0
1
0 7 8 5
1
end_operator
begin_operator
turn_to satellite0 planet9 star5
0
1
0 7 9 5
1
end_operator
begin_operator
turn_to satellite0 planet9 star6
0
1
0 7 10 5
1
end_operator
begin_operator
turn_to satellite0 planet9 star7
0
1
0 7 11 5
1
end_operator
begin_operator
turn_to satellite0 star1 groundstation0
0
1
0 7 0 6
1
end_operator
begin_operator
turn_to satellite0 star1 groundstation3
0
1
0 7 1 6
1
end_operator
begin_operator
turn_to satellite0 star1 groundstation4
0
1
0 7 2 6
1
end_operator
begin_operator
turn_to satellite0 star1 phenomenon8
0
1
0 7 3 6
1
end_operator
begin_operator
turn_to satellite0 star1 planet10
0
1
0 7 4 6
1
end_operator
begin_operator
turn_to satellite0 star1 planet9
0
1
0 7 5 6
1
end_operator
begin_operator
turn_to satellite0 star1 star11
0
1
0 7 7 6
1
end_operator
begin_operator
turn_to satellite0 star1 star2
0
1
0 7 8 6
1
end_operator
begin_operator
turn_to satellite0 star1 star5
0
1
0 7 9 6
1
end_operator
begin_operator
turn_to satellite0 star1 star6
0
1
0 7 10 6
1
end_operator
begin_operator
turn_to satellite0 star1 star7
0
1
0 7 11 6
1
end_operator
begin_operator
turn_to satellite0 star11 groundstation0
0
1
0 7 0 7
1
end_operator
begin_operator
turn_to satellite0 star11 groundstation3
0
1
0 7 1 7
1
end_operator
begin_operator
turn_to satellite0 star11 groundstation4
0
1
0 7 2 7
1
end_operator
begin_operator
turn_to satellite0 star11 phenomenon8
0
1
0 7 3 7
1
end_operator
begin_operator
turn_to satellite0 star11 planet10
0
1
0 7 4 7
1
end_operator
begin_operator
turn_to satellite0 star11 planet9
0
1
0 7 5 7
1
end_operator
begin_operator
turn_to satellite0 star11 star1
0
1
0 7 6 7
1
end_operator
begin_operator
turn_to satellite0 star11 star2
0
1
0 7 8 7
1
end_operator
begin_operator
turn_to satellite0 star11 star5
0
1
0 7 9 7
1
end_operator
begin_operator
turn_to satellite0 star11 star6
0
1
0 7 10 7
1
end_operator
begin_operator
turn_to satellite0 star11 star7
0
1
0 7 11 7
1
end_operator
begin_operator
turn_to satellite0 star2 groundstation0
0
1
0 7 0 8
1
end_operator
begin_operator
turn_to satellite0 star2 groundstation3
0
1
0 7 1 8
1
end_operator
begin_operator
turn_to satellite0 star2 groundstation4
0
1
0 7 2 8
1
end_operator
begin_operator
turn_to satellite0 star2 phenomenon8
0
1
0 7 3 8
1
end_operator
begin_operator
turn_to satellite0 star2 planet10
0
1
0 7 4 8
1
end_operator
begin_operator
turn_to satellite0 star2 planet9
0
1
0 7 5 8
1
end_operator
begin_operator
turn_to satellite0 star2 star1
0
1
0 7 6 8
1
end_operator
begin_operator
turn_to satellite0 star2 star11
0
1
0 7 7 8
1
end_operator
begin_operator
turn_to satellite0 star2 star5
0
1
0 7 9 8
1
end_operator
begin_operator
turn_to satellite0 star2 star6
0
1
0 7 10 8
1
end_operator
begin_operator
turn_to satellite0 star2 star7
0
1
0 7 11 8
1
end_operator
begin_operator
turn_to satellite0 star5 groundstation0
0
1
0 7 0 9
1
end_operator
begin_operator
turn_to satellite0 star5 groundstation3
0
1
0 7 1 9
1
end_operator
begin_operator
turn_to satellite0 star5 groundstation4
0
1
0 7 2 9
1
end_operator
begin_operator
turn_to satellite0 star5 phenomenon8
0
1
0 7 3 9
1
end_operator
begin_operator
turn_to satellite0 star5 planet10
0
1
0 7 4 9
1
end_operator
begin_operator
turn_to satellite0 star5 planet9
0
1
0 7 5 9
1
end_operator
begin_operator
turn_to satellite0 star5 star1
0
1
0 7 6 9
1
end_operator
begin_operator
turn_to satellite0 star5 star11
0
1
0 7 7 9
1
end_operator
begin_operator
turn_to satellite0 star5 star2
0
1
0 7 8 9
1
end_operator
begin_operator
turn_to satellite0 star5 star6
0
1
0 7 10 9
1
end_operator
begin_operator
turn_to satellite0 star5 star7
0
1
0 7 11 9
1
end_operator
begin_operator
turn_to satellite0 star6 groundstation0
0
1
0 7 0 10
1
end_operator
begin_operator
turn_to satellite0 star6 groundstation3
0
1
0 7 1 10
1
end_operator
begin_operator
turn_to satellite0 star6 groundstation4
0
1
0 7 2 10
1
end_operator
begin_operator
turn_to satellite0 star6 phenomenon8
0
1
0 7 3 10
1
end_operator
begin_operator
turn_to satellite0 star6 planet10
0
1
0 7 4 10
1
end_operator
begin_operator
turn_to satellite0 star6 planet9
0
1
0 7 5 10
1
end_operator
begin_operator
turn_to satellite0 star6 star1
0
1
0 7 6 10
1
end_operator
begin_operator
turn_to satellite0 star6 star11
0
1
0 7 7 10
1
end_operator
begin_operator
turn_to satellite0 star6 star2
0
1
0 7 8 10
1
end_operator
begin_operator
turn_to satellite0 star6 star5
0
1
0 7 9 10
1
end_operator
begin_operator
turn_to satellite0 star6 star7
0
1
0 7 11 10
1
end_operator
begin_operator
turn_to satellite0 star7 groundstation0
0
1
0 7 0 11
1
end_operator
begin_operator
turn_to satellite0 star7 groundstation3
0
1
0 7 1 11
1
end_operator
begin_operator
turn_to satellite0 star7 groundstation4
0
1
0 7 2 11
1
end_operator
begin_operator
turn_to satellite0 star7 phenomenon8
0
1
0 7 3 11
1
end_operator
begin_operator
turn_to satellite0 star7 planet10
0
1
0 7 4 11
1
end_operator
begin_operator
turn_to satellite0 star7 planet9
0
1
0 7 5 11
1
end_operator
begin_operator
turn_to satellite0 star7 star1
0
1
0 7 6 11
1
end_operator
begin_operator
turn_to satellite0 star7 star11
0
1
0 7 7 11
1
end_operator
begin_operator
turn_to satellite0 star7 star2
0
1
0 7 8 11
1
end_operator
begin_operator
turn_to satellite0 star7 star5
0
1
0 7 9 11
1
end_operator
begin_operator
turn_to satellite0 star7 star6
0
1
0 7 10 11
1
end_operator
begin_operator
turn_to satellite1 groundstation0 groundstation3
0
1
0 6 1 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 groundstation4
0
1
0 6 2 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 phenomenon8
0
1
0 6 3 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 planet10
0
1
0 6 4 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 planet9
0
1
0 6 5 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 star1
0
1
0 6 6 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 star11
0
1
0 6 7 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 star2
0
1
0 6 8 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 star5
0
1
0 6 9 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 star6
0
1
0 6 10 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 star7
0
1
0 6 11 0
1
end_operator
begin_operator
turn_to satellite1 groundstation3 groundstation0
0
1
0 6 0 1
1
end_operator
begin_operator
turn_to satellite1 groundstation3 groundstation4
0
1
0 6 2 1
1
end_operator
begin_operator
turn_to satellite1 groundstation3 phenomenon8
0
1
0 6 3 1
1
end_operator
begin_operator
turn_to satellite1 groundstation3 planet10
0
1
0 6 4 1
1
end_operator
begin_operator
turn_to satellite1 groundstation3 planet9
0
1
0 6 5 1
1
end_operator
begin_operator
turn_to satellite1 groundstation3 star1
0
1
0 6 6 1
1
end_operator
begin_operator
turn_to satellite1 groundstation3 star11
0
1
0 6 7 1
1
end_operator
begin_operator
turn_to satellite1 groundstation3 star2
0
1
0 6 8 1
1
end_operator
begin_operator
turn_to satellite1 groundstation3 star5
0
1
0 6 9 1
1
end_operator
begin_operator
turn_to satellite1 groundstation3 star6
0
1
0 6 10 1
1
end_operator
begin_operator
turn_to satellite1 groundstation3 star7
0
1
0 6 11 1
1
end_operator
begin_operator
turn_to satellite1 groundstation4 groundstation0
0
1
0 6 0 2
1
end_operator
begin_operator
turn_to satellite1 groundstation4 groundstation3
0
1
0 6 1 2
1
end_operator
begin_operator
turn_to satellite1 groundstation4 phenomenon8
0
1
0 6 3 2
1
end_operator
begin_operator
turn_to satellite1 groundstation4 planet10
0
1
0 6 4 2
1
end_operator
begin_operator
turn_to satellite1 groundstation4 planet9
0
1
0 6 5 2
1
end_operator
begin_operator
turn_to satellite1 groundstation4 star1
0
1
0 6 6 2
1
end_operator
begin_operator
turn_to satellite1 groundstation4 star11
0
1
0 6 7 2
1
end_operator
begin_operator
turn_to satellite1 groundstation4 star2
0
1
0 6 8 2
1
end_operator
begin_operator
turn_to satellite1 groundstation4 star5
0
1
0 6 9 2
1
end_operator
begin_operator
turn_to satellite1 groundstation4 star6
0
1
0 6 10 2
1
end_operator
begin_operator
turn_to satellite1 groundstation4 star7
0
1
0 6 11 2
1
end_operator
begin_operator
turn_to satellite1 phenomenon8 groundstation0
0
1
0 6 0 3
1
end_operator
begin_operator
turn_to satellite1 phenomenon8 groundstation3
0
1
0 6 1 3
1
end_operator
begin_operator
turn_to satellite1 phenomenon8 groundstation4
0
1
0 6 2 3
1
end_operator
begin_operator
turn_to satellite1 phenomenon8 planet10
0
1
0 6 4 3
1
end_operator
begin_operator
turn_to satellite1 phenomenon8 planet9
0
1
0 6 5 3
1
end_operator
begin_operator
turn_to satellite1 phenomenon8 star1
0
1
0 6 6 3
1
end_operator
begin_operator
turn_to satellite1 phenomenon8 star11
0
1
0 6 7 3
1
end_operator
begin_operator
turn_to satellite1 phenomenon8 star2
0
1
0 6 8 3
1
end_operator
begin_operator
turn_to satellite1 phenomenon8 star5
0
1
0 6 9 3
1
end_operator
begin_operator
turn_to satellite1 phenomenon8 star6
0
1
0 6 10 3
1
end_operator
begin_operator
turn_to satellite1 phenomenon8 star7
0
1
0 6 11 3
1
end_operator
begin_operator
turn_to satellite1 planet10 groundstation0
0
1
0 6 0 4
1
end_operator
begin_operator
turn_to satellite1 planet10 groundstation3
0
1
0 6 1 4
1
end_operator
begin_operator
turn_to satellite1 planet10 groundstation4
0
1
0 6 2 4
1
end_operator
begin_operator
turn_to satellite1 planet10 phenomenon8
0
1
0 6 3 4
1
end_operator
begin_operator
turn_to satellite1 planet10 planet9
0
1
0 6 5 4
1
end_operator
begin_operator
turn_to satellite1 planet10 star1
0
1
0 6 6 4
1
end_operator
begin_operator
turn_to satellite1 planet10 star11
0
1
0 6 7 4
1
end_operator
begin_operator
turn_to satellite1 planet10 star2
0
1
0 6 8 4
1
end_operator
begin_operator
turn_to satellite1 planet10 star5
0
1
0 6 9 4
1
end_operator
begin_operator
turn_to satellite1 planet10 star6
0
1
0 6 10 4
1
end_operator
begin_operator
turn_to satellite1 planet10 star7
0
1
0 6 11 4
1
end_operator
begin_operator
turn_to satellite1 planet9 groundstation0
0
1
0 6 0 5
1
end_operator
begin_operator
turn_to satellite1 planet9 groundstation3
0
1
0 6 1 5
1
end_operator
begin_operator
turn_to satellite1 planet9 groundstation4
0
1
0 6 2 5
1
end_operator
begin_operator
turn_to satellite1 planet9 phenomenon8
0
1
0 6 3 5
1
end_operator
begin_operator
turn_to satellite1 planet9 planet10
0
1
0 6 4 5
1
end_operator
begin_operator
turn_to satellite1 planet9 star1
0
1
0 6 6 5
1
end_operator
begin_operator
turn_to satellite1 planet9 star11
0
1
0 6 7 5
1
end_operator
begin_operator
turn_to satellite1 planet9 star2
0
1
0 6 8 5
1
end_operator
begin_operator
turn_to satellite1 planet9 star5
0
1
0 6 9 5
1
end_operator
begin_operator
turn_to satellite1 planet9 star6
0
1
0 6 10 5
1
end_operator
begin_operator
turn_to satellite1 planet9 star7
0
1
0 6 11 5
1
end_operator
begin_operator
turn_to satellite1 star1 groundstation0
0
1
0 6 0 6
1
end_operator
begin_operator
turn_to satellite1 star1 groundstation3
0
1
0 6 1 6
1
end_operator
begin_operator
turn_to satellite1 star1 groundstation4
0
1
0 6 2 6
1
end_operator
begin_operator
turn_to satellite1 star1 phenomenon8
0
1
0 6 3 6
1
end_operator
begin_operator
turn_to satellite1 star1 planet10
0
1
0 6 4 6
1
end_operator
begin_operator
turn_to satellite1 star1 planet9
0
1
0 6 5 6
1
end_operator
begin_operator
turn_to satellite1 star1 star11
0
1
0 6 7 6
1
end_operator
begin_operator
turn_to satellite1 star1 star2
0
1
0 6 8 6
1
end_operator
begin_operator
turn_to satellite1 star1 star5
0
1
0 6 9 6
1
end_operator
begin_operator
turn_to satellite1 star1 star6
0
1
0 6 10 6
1
end_operator
begin_operator
turn_to satellite1 star1 star7
0
1
0 6 11 6
1
end_operator
begin_operator
turn_to satellite1 star11 groundstation0
0
1
0 6 0 7
1
end_operator
begin_operator
turn_to satellite1 star11 groundstation3
0
1
0 6 1 7
1
end_operator
begin_operator
turn_to satellite1 star11 groundstation4
0
1
0 6 2 7
1
end_operator
begin_operator
turn_to satellite1 star11 phenomenon8
0
1
0 6 3 7
1
end_operator
begin_operator
turn_to satellite1 star11 planet10
0
1
0 6 4 7
1
end_operator
begin_operator
turn_to satellite1 star11 planet9
0
1
0 6 5 7
1
end_operator
begin_operator
turn_to satellite1 star11 star1
0
1
0 6 6 7
1
end_operator
begin_operator
turn_to satellite1 star11 star2
0
1
0 6 8 7
1
end_operator
begin_operator
turn_to satellite1 star11 star5
0
1
0 6 9 7
1
end_operator
begin_operator
turn_to satellite1 star11 star6
0
1
0 6 10 7
1
end_operator
begin_operator
turn_to satellite1 star11 star7
0
1
0 6 11 7
1
end_operator
begin_operator
turn_to satellite1 star2 groundstation0
0
1
0 6 0 8
1
end_operator
begin_operator
turn_to satellite1 star2 groundstation3
0
1
0 6 1 8
1
end_operator
begin_operator
turn_to satellite1 star2 groundstation4
0
1
0 6 2 8
1
end_operator
begin_operator
turn_to satellite1 star2 phenomenon8
0
1
0 6 3 8
1
end_operator
begin_operator
turn_to satellite1 star2 planet10
0
1
0 6 4 8
1
end_operator
begin_operator
turn_to satellite1 star2 planet9
0
1
0 6 5 8
1
end_operator
begin_operator
turn_to satellite1 star2 star1
0
1
0 6 6 8
1
end_operator
begin_operator
turn_to satellite1 star2 star11
0
1
0 6 7 8
1
end_operator
begin_operator
turn_to satellite1 star2 star5
0
1
0 6 9 8
1
end_operator
begin_operator
turn_to satellite1 star2 star6
0
1
0 6 10 8
1
end_operator
begin_operator
turn_to satellite1 star2 star7
0
1
0 6 11 8
1
end_operator
begin_operator
turn_to satellite1 star5 groundstation0
0
1
0 6 0 9
1
end_operator
begin_operator
turn_to satellite1 star5 groundstation3
0
1
0 6 1 9
1
end_operator
begin_operator
turn_to satellite1 star5 groundstation4
0
1
0 6 2 9
1
end_operator
begin_operator
turn_to satellite1 star5 phenomenon8
0
1
0 6 3 9
1
end_operator
begin_operator
turn_to satellite1 star5 planet10
0
1
0 6 4 9
1
end_operator
begin_operator
turn_to satellite1 star5 planet9
0
1
0 6 5 9
1
end_operator
begin_operator
turn_to satellite1 star5 star1
0
1
0 6 6 9
1
end_operator
begin_operator
turn_to satellite1 star5 star11
0
1
0 6 7 9
1
end_operator
begin_operator
turn_to satellite1 star5 star2
0
1
0 6 8 9
1
end_operator
begin_operator
turn_to satellite1 star5 star6
0
1
0 6 10 9
1
end_operator
begin_operator
turn_to satellite1 star5 star7
0
1
0 6 11 9
1
end_operator
begin_operator
turn_to satellite1 star6 groundstation0
0
1
0 6 0 10
1
end_operator
begin_operator
turn_to satellite1 star6 groundstation3
0
1
0 6 1 10
1
end_operator
begin_operator
turn_to satellite1 star6 groundstation4
0
1
0 6 2 10
1
end_operator
begin_operator
turn_to satellite1 star6 phenomenon8
0
1
0 6 3 10
1
end_operator
begin_operator
turn_to satellite1 star6 planet10
0
1
0 6 4 10
1
end_operator
begin_operator
turn_to satellite1 star6 planet9
0
1
0 6 5 10
1
end_operator
begin_operator
turn_to satellite1 star6 star1
0
1
0 6 6 10
1
end_operator
begin_operator
turn_to satellite1 star6 star11
0
1
0 6 7 10
1
end_operator
begin_operator
turn_to satellite1 star6 star2
0
1
0 6 8 10
1
end_operator
begin_operator
turn_to satellite1 star6 star5
0
1
0 6 9 10
1
end_operator
begin_operator
turn_to satellite1 star6 star7
0
1
0 6 11 10
1
end_operator
begin_operator
turn_to satellite1 star7 groundstation0
0
1
0 6 0 11
1
end_operator
begin_operator
turn_to satellite1 star7 groundstation3
0
1
0 6 1 11
1
end_operator
begin_operator
turn_to satellite1 star7 groundstation4
0
1
0 6 2 11
1
end_operator
begin_operator
turn_to satellite1 star7 phenomenon8
0
1
0 6 3 11
1
end_operator
begin_operator
turn_to satellite1 star7 planet10
0
1
0 6 4 11
1
end_operator
begin_operator
turn_to satellite1 star7 planet9
0
1
0 6 5 11
1
end_operator
begin_operator
turn_to satellite1 star7 star1
0
1
0 6 6 11
1
end_operator
begin_operator
turn_to satellite1 star7 star11
0
1
0 6 7 11
1
end_operator
begin_operator
turn_to satellite1 star7 star2
0
1
0 6 8 11
1
end_operator
begin_operator
turn_to satellite1 star7 star5
0
1
0 6 9 11
1
end_operator
begin_operator
turn_to satellite1 star7 star6
0
1
0 6 10 11
1
end_operator
0
