begin_version
3
end_version
begin_metric
0
end_metric
28
begin_variable
var0
-1
2
Atom calibrated(instrument0)
NegatedAtom calibrated(instrument0)
end_variable
begin_variable
var1
-1
2
Atom calibrated(instrument1)
NegatedAtom calibrated(instrument1)
end_variable
begin_variable
var2
-1
2
Atom calibrated(instrument2)
NegatedAtom calibrated(instrument2)
end_variable
begin_variable
var3
-1
2
Atom calibrated(instrument3)
NegatedAtom calibrated(instrument3)
end_variable
begin_variable
var4
-1
2
Atom have_image(groundstation0, infrared1)
NegatedAtom have_image(groundstation0, infrared1)
end_variable
begin_variable
var5
-1
2
Atom have_image(groundstation0, thermograph0)
NegatedAtom have_image(groundstation0, thermograph0)
end_variable
begin_variable
var6
-1
2
Atom have_image(groundstation1, infrared1)
NegatedAtom have_image(groundstation1, infrared1)
end_variable
begin_variable
var7
-1
2
Atom have_image(groundstation1, thermograph0)
NegatedAtom have_image(groundstation1, thermograph0)
end_variable
begin_variable
var8
-1
2
Atom have_image(groundstation2, infrared1)
NegatedAtom have_image(groundstation2, infrared1)
end_variable
begin_variable
var9
-1
2
Atom have_image(groundstation2, thermograph0)
NegatedAtom have_image(groundstation2, thermograph0)
end_variable
begin_variable
var10
-1
2
Atom have_image(groundstation3, infrared1)
NegatedAtom have_image(groundstation3, infrared1)
end_variable
begin_variable
var11
-1
2
Atom have_image(groundstation3, thermograph0)
NegatedAtom have_image(groundstation3, thermograph0)
end_variable
begin_variable
var12
-1
2
Atom have_image(phenomenon7, infrared1)
NegatedAtom have_image(phenomenon7, infrared1)
end_variable
begin_variable
var13
-1
2
Atom have_image(phenomenon7, thermograph0)
NegatedAtom have_image(phenomenon7, thermograph0)
end_variable
begin_variable
var14
-1
2
Atom have_image(planet5, infrared1)
NegatedAtom have_image(planet5, infrared1)
end_variable
begin_variable
var15
-1
2
Atom have_image(planet5, thermograph0)
NegatedAtom have_image(planet5, thermograph0)
end_variable
begin_variable
var16
-1
2
Atom have_image(star4, infrared1)
NegatedAtom have_image(star4, infrared1)
end_variable
begin_variable
var17
-1
2
Atom have_image(star4, thermograph0)
NegatedAtom have_image(star4, thermograph0)
end_variable
begin_variable
var18
-1
2
Atom have_image(star6, infrared1)
NegatedAtom have_image(star6, infrared1)
end_variable
begin_variable
var19
-1
2
Atom have_image(star6, thermograph0)
NegatedAtom have_image(star6, thermograph0)
end_variable
begin_variable
var20
-1
8
Atom pointing(satellite0, groundstation0)
Atom pointing(satellite0, groundstation1)
Atom pointing(satellite0, groundstation2)
Atom pointing(satellite0, groundstation3)
Atom pointing(satellite0, phenomenon7)
Atom pointing(satellite0, planet5)
Atom pointing(satellite0, star4)
Atom pointing(satellite0, star6)
end_variable
begin_variable
var21
-1
8
Atom pointing(satellite1, groundstation0)
Atom pointing(satellite1, groundstation1)
Atom pointing(satellite1, groundstation2)
Atom pointing(satellite1, groundstation3)
Atom pointing(satellite1, phenomenon7)
Atom pointing(satellite1, planet5)
Atom pointing(satellite1, star4)
Atom pointing(satellite1, star6)
end_variable
begin_variable
var22
-1
2
Atom power_avail(satellite0)
NegatedAtom power_avail(satellite0)
end_variable
begin_variable
var23
-1
2
Atom power_avail(satellite1)
NegatedAtom power_avail(satellite1)
end_variable
begin_variable
var24
-1
2
Atom power_on(instrument0)
NegatedAtom power_on(instrument0)
end_variable
begin_variable
var25
-1
2
Atom power_on(instrument1)
NegatedAtom power_on(instrument1)
end_variable
begin_variable
var26
-1
2
Atom power_on(instrument2)
NegatedAtom power_on(instrument2)
end_variable
begin_variable
var27
-1
2
Atom power_on(instrument3)
NegatedAtom power_on(instrument3)
end_variable
2
begin_mutex_group
8
20 0
20 1
20 2
20 3
20 4
20 5
20 6
20 7
end_mutex_group
begin_mutex_group
8
21 0
21 1
21 2
21 3
21 4
21 5
21 6
21 7
end_mutex_group
begin_state
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
5
5
0
0
1
1
1
1
end_state
begin_goal
4
12 0
14 0
18 0
21 5
end_goal
180
begin_operator
calibrate satellite0 instrument0 groundstation1
2
20 1
24 0
1
0 0 -1 0
1
end_operator
begin_operator
calibrate satellite0 instrument1 groundstation3
2
20 3
25 0
1
0 1 -1 0
1
end_operator
begin_operator
calibrate satellite1 instrument2 groundstation0
2
21 0
26 0
1
0 2 -1 0
1
end_operator
begin_operator
calibrate satellite1 instrument3 groundstation0
2
21 0
27 0
1
0 3 -1 0
1
end_operator
begin_operator
switch_off instrument0 satellite0
0
2
0 22 -1 0
0 24 0 1
1
end_operator
begin_operator
switch_off instrument1 satellite0
0
2
0 22 -1 0
0 25 0 1
1
end_operator
begin_operator
switch_off instrument2 satellite1
0
2
0 23 -1 0
0 26 0 1
1
end_operator
begin_operator
switch_off instrument3 satellite1
0
2
0 23 -1 0
0 27 0 1
1
end_operator
begin_operator
switch_on instrument0 satellite0
0
3
0 0 -1 1
0 22 0 1
0 24 -1 0
1
end_operator
begin_operator
switch_on instrument1 satellite0
0
3
0 1 -1 1
0 22 0 1
0 25 -1 0
1
end_operator
begin_operator
switch_on instrument2 satellite1
0
3
0 2 -1 1
0 23 0 1
0 26 -1 0
1
end_operator
begin_operator
switch_on instrument3 satellite1
0
3
0 3 -1 1
0 23 0 1
0 27 -1 0
1
end_operator
begin_operator
take_image satellite0 groundstation0 instrument0 infrared1
3
0 0
20 0
24 0
1
0 4 -1 0
1
end_operator
begin_operator
take_image satellite0 groundstation0 instrument0 thermograph0
3
0 0
20 0
24 0
1
0 5 -1 0
1
end_operator
begin_operator
take_image satellite0 groundstation0 instrument1 infrared1
3
1 0
20 0
25 0
1
0 4 -1 0
1
end_operator
begin_operator
take_image satellite0 groundstation0 instrument1 thermograph0
3
1 0
20 0
25 0
1
0 5 -1 0
1
end_operator
begin_operator
take_image satellite0 groundstation1 instrument0 infrared1
3
0 0
20 1
24 0
1
0 6 -1 0
1
end_operator
begin_operator
take_image satellite0 groundstation1 instrument0 thermograph0
3
0 0
20 1
24 0
1
0 7 -1 0
1
end_operator
begin_operator
take_image satellite0 groundstation1 instrument1 infrared1
3
1 0
20 1
25 0
1
0 6 -1 0
1
end_operator
begin_operator
take_image satellite0 groundstation1 instrument1 thermograph0
3
1 0
20 1
25 0
1
0 7 -1 0
1
end_operator
begin_operator
take_image satellite0 groundstation2 instrument0 infrared1
3
0 0
20 2
24 0
1
0 8 -1 0
1
end_operator
begin_operator
take_image satellite0 groundstation2 instrument0 thermograph0
3
0 0
20 2
24 0
1
0 9 -1 0
1
end_operator
begin_operator
take_image satellite0 groundstation2 instrument1 infrared1
3
1 0
20 2
25 0
1
0 8 -1 0
1
end_operator
begin_operator
take_image satellite0 groundstation2 instrument1 thermograph0
3
1 0
20 2
25 0
1
0 9 -1 0
1
end_operator
begin_operator
take_image satellite0 groundstation3 instrument0 infrared1
3
0 0
20 3
24 0
1
0 10 -1 0
1
end_operator
begin_operator
take_image satellite0 groundstation3 instrument0 thermograph0
3
0 0
20 3
24 0
1
0 11 -1 0
1
end_operator
begin_operator
take_image satellite0 groundstation3 instrument1 infrared1
3
1 0
20 3
25 0
1
0 10 -1 0
1
end_operator
begin_operator
take_image satellite0 groundstation3 instrument1 thermograph0
3
1 0
20 3
25 0
1
0 11 -1 0
1
end_operator
begin_operator
take_image satellite0 phenomenon7 instrument0 infrared1
3
0 0
20 4
24 0
1
0 12 -1 0
1
end_operator
begin_operator
take_image satellite0 phenomenon7 instrument0 thermograph0
3
0 0
20 4
24 0
1
0 13 -1 0
1
end_operator
begin_operator
take_image satellite0 phenomenon7 instrument1 infrared1
3
1 0
20 4
25 0
1
0 12 -1 0
1
end_operator
begin_operator
take_image satellite0 phenomenon7 instrument1 thermograph0
3
1 0
20 4
25 0
1
0 13 -1 0
1
end_operator
begin_operator
take_image satellite0 planet5 instrument0 infrared1
3
0 0
20 5
24 0
1
0 14 -1 0
1
end_operator
begin_operator
take_image satellite0 planet5 instrument0 thermograph0
3
0 0
20 5
24 0
1
0 15 -1 0
1
end_operator
begin_operator
take_image satellite0 planet5 instrument1 infrared1
3
1 0
20 5
25 0
1
0 14 -1 0
1
end_operator
begin_operator
take_image satellite0 planet5 instrument1 thermograph0
3
1 0
20 5
25 0
1
0 15 -1 0
1
end_operator
begin_operator
take_image satellite0 star4 instrument0 infrared1
3
0 0
20 6
24 0
1
0 16 -1 0
1
end_operator
begin_operator
take_image satellite0 star4 instrument0 thermograph0
3
0 0
20 6
24 0
1
0 17 -1 0
1
end_operator
begin_operator
take_image satellite0 star4 instrument1 infrared1
3
1 0
20 6
25 0
1
0 16 -1 0
1
end_operator
begin_operator
take_image satellite0 star4 instrument1 thermograph0
3
1 0
20 6
25 0
1
0 17 -1 0
1
end_operator
begin_operator
take_image satellite0 star6 instrument0 infrared1
3
0 0
20 7
24 0
1
0 18 -1 0
1
end_operator
begin_operator
take_image satellite0 star6 instrument0 thermograph0
3
0 0
20 7
24 0
1
0 19 -1 0
1
end_operator
begin_operator
take_image satellite0 star6 instrument1 infrared1
3
1 0
20 7
25 0
1
0 18 -1 0
1
end_operator
begin_operator
take_image satellite0 star6 instrument1 thermograph0
3
1 0
20 7
25 0
1
0 19 -1 0
1
end_operator
begin_operator
take_image satellite1 groundstation0 instrument2 infrared1
3
2 0
21 0
26 0
1
0 4 -1 0
1
end_operator
begin_operator
take_image satellite1 groundstation0 instrument2 thermograph0
3
2 0
21 0
26 0
1
0 5 -1 0
1
end_operator
begin_operator
take_image satellite1 groundstation0 instrument3 thermograph0
3
3 0
21 0
27 0
1
0 5 -1 0
1
end_operator
begin_operator
take_image satellite1 groundstation1 instrument2 infrared1
3
2 0
21 1
26 0
1
0 6 -1 0
1
end_operator
begin_operator
take_image satellite1 groundstation1 instrument2 thermograph0
3
2 0
21 1
26 0
1
0 7 -1 0
1
end_operator
begin_operator
take_image satellite1 groundstation1 instrument3 thermograph0
3
3 0
21 1
27 0
1
0 7 -1 0
1
end_operator
begin_operator
take_image satellite1 groundstation2 instrument2 infrared1
3
2 0
21 2
26 0
1
0 8 -1 0
1
end_operator
begin_operator
take_image satellite1 groundstation2 instrument2 thermograph0
3
2 0
21 2
26 0
1
0 9 -1 0
1
end_operator
begin_operator
take_image satellite1 groundstation2 instrument3 thermograph0
3
3 0
21 2
27 0
1
0 9 -1 0
1
end_operator
begin_operator
take_image satellite1 groundstation3 instrument2 infrared1
3
2 0
21 3
26 0
1
0 10 -1 0
1
end_operator
begin_operator
take_image satellite1 groundstation3 instrument2 thermograph0
3
2 0
21 3
26 0
1
0 11 -1 0
1
end_operator
begin_operator
take_image satellite1 groundstation3 instrument3 thermograph0
3
3 0
21 3
27 0
1
0 11 -1 0
1
end_operator
begin_operator
take_image satellite1 phenomenon7 instrument2 infrared1
3
2 0
21 4
26 0
1
0 12 -1 0
1
end_operator
begin_operator
take_image satellite1 phenomenon7 instrument2 thermograph0
3
2 0
21 4
26 0
1
0 13 -1 0
1
end_operator
begin_operator
take_image satellite1 phenomenon7 instrument3 thermograph0
3
3 0
21 4
27 0
1
0 13 -1 0
1
end_operator
begin_operator
take_image satellite1 planet5 instrument2 infrared1
3
2 0
21 5
26 0
1
0 14 -1 0
1
end_operator
begin_operator
take_image satellite1 planet5 instrument2 thermograph0
3
2 0
21 5
26 0
1
0 15 -1 0
1
end_operator
begin_operator
take_image satellite1 planet5 instrument3 thermograph0
3
3 0
21 5
27 0
1
0 15 -1 0
1
end_operator
begin_operator
take_image satellite1 star4 instrument2 infrared1
3
2 0
21 6
26 0
1
0 16 -1 0
1
end_operator
begin_operator
take_image satellite1 star4 instrument2 thermograph0
3
2 0
21 6
26 0
1
0 17 -1 0
1
end_operator
begin_operator
take_image satellite1 star4 instrument3 thermograph0
3
3 0
21 6
27 0
1
0 17 -1 0
1
end_operator
begin_operator
take_image satellite1 star6 instrument2 infrared1
3
2 0
21 7
26 0
1
0 18 -1 0
1
end_operator
begin_operator
take_image satellite1 star6 instrument2 thermograph0
3
2 0
21 7
26 0
1
0 19 -1 0
1
end_operator
begin_operator
take_image satellite1 star6 instrument3 thermograph0
3
3 0
21 7
27 0
1
0 19 -1 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 groundstation1
0
1
0 20 1 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 groundstation2
0
1
0 20 2 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 groundstation3
0
1
0 20 3 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 phenomenon7
0
1
0 20 4 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 planet5
0
1
0 20 5 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 star4
0
1
0 20 6 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 star6
0
1
0 20 7 0
1
end_operator
begin_operator
turn_to satellite0 groundstation1 groundstation0
0
1
0 20 0 1
1
end_operator
begin_operator
turn_to satellite0 groundstation1 groundstation2
0
1
0 20 2 1
1
end_operator
begin_operator
turn_to satellite0 groundstation1 groundstation3
0
1
0 20 3 1
1
end_operator
begin_operator
turn_to satellite0 groundstation1 phenomenon7
0
1
0 20 4 1
1
end_operator
begin_operator
turn_to satellite0 groundstation1 planet5
0
1
0 20 5 1
1
end_operator
begin_operator
turn_to satellite0 groundstation1 star4
0
1
0 20 6 1
1
end_operator
begin_operator
turn_to satellite0 groundstation1 star6
0
1
0 20 7 1
1
end_operator
begin_operator
turn_to satellite0 groundstation2 groundstation0
0
1
0 20 0 2
1
end_operator
begin_operator
turn_to satellite0 groundstation2 groundstation1
0
1
0 20 1 2
1
end_operator
begin_operator
turn_to satellite0 groundstation2 groundstation3
0
1
0 20 3 2
1
end_operator
begin_operator
turn_to satellite0 groundstation2 phenomenon7
0
1
0 20 4 2
1
end_operator
begin_operator
turn_to satellite0 groundstation2 planet5
0
1
0 20 5 2
1
end_operator
begin_operator
turn_to satellite0 groundstation2 star4
0
1
0 20 6 2
1
end_operator
begin_operator
turn_to satellite0 groundstation2 star6
0
1
0 20 7 2
1
end_operator
begin_operator
turn_to satellite0 groundstation3 groundstation0
0
1
0 20 0 3
1
end_operator
begin_operator
turn_to satellite0 groundstation3 groundstation1
0
1
0 20 1 3
1
end_operator
begin_operator
turn_to satellite0 groundstation3 groundstation2
0
1
0 20 2 3
1
end_operator
begin_operator
turn_to satellite0 groundstation3 phenomenon7
0
1
0 20 4 3
1
end_operator
begin_operator
turn_to satellite0 groundstation3 planet5
0
1
0 20 5 3
1
end_operator
begin_operator
turn_to satellite0 groundstation3 star4
0
1
0 20 6 3
1
end_operator
begin_operator
turn_to satellite0 groundstation3 star6
0
1
0 20 7 3
1
end_operator
begin_operator
turn_to satellite0 phenomenon7 groundstation0
0
1
0 20 0 4
1
end_operator
begin_operator
turn_to satellite0 phenomenon7 groundstation1
0
1
0 20 1 4
1
end_operator
begin_operator
turn_to satellite0 phenomenon7 groundstation2
0
1
0 20 2 4
1
end_operator
begin_operator
turn_to satellite0 phenomenon7 groundstation3
0
1
0 20 3 4
1
end_operator
begin_operator
turn_to satellite0 phenomenon7 planet5
0
1
0 20 5 4
1
end_operator
begin_operator
turn_to satellite0 phenomenon7 star4
0
1
0 20 6 4
1
end_operator
begin_operator
turn_to satellite0 phenomenon7 star6
0
1
0 20 7 4
1
end_operator
begin_operator
turn_to satellite0 planet5 groundstation0
0
1
0 20 0 5
1
end_operator
begin_operator
turn_to satellite0 planet5 groundstation1
0
1
0 20 1 5
1
end_operator
begin_operator
turn_to satellite0 planet5 groundstation2
0
1
0 20 2 5
1
end_operator
begin_operator
turn_to satellite0 planet5 groundstation3
0
1
0 20 3 5
1
end_operator
begin_operator
turn_to satellite0 planet5 phenomenon7
0
1
0 20 4 5
1
end_operator
begin_operator
turn_to satellite0 planet5 star4
0
1
0 20 6 5
1
end_operator
begin_operator
turn_to satellite0 planet5 star6
0
1
0 20 7 5
1
end_operator
begin_operator
turn_to satellite0 star4 groundstation0
0
1
0 20 0 6
1
end_operator
begin_operator
turn_to satellite0 star4 groundstation1
0
1
0 20 1 6
1
end_operator
begin_operator
turn_to satellite0 star4 groundstation2
0
1
0 20 2 6
1
end_operator
begin_operator
turn_to satellite0 star4 groundstation3
0
1
0 20 3 6
1
end_operator
begin_operator
turn_to satellite0 star4 phenomenon7
0
1
0 20 4 6
1
end_operator
begin_operator
turn_to satellite0 star4 planet5
0
1
0 20 5 6
1
end_operator
begin_operator
turn_to satellite0 star4 star6
0
1
0 20 7 6
1
end_operator
begin_operator
turn_to satellite0 star6 groundstation0
0
1
0 20 0 7
1
end_operator
begin_operator
turn_to satellite0 star6 groundstation1
0
1
0 20 1 7
1
end_operator
begin_operator
turn_to satellite0 star6 groundstation2
0
1
0 20 2 7
1
end_operator
begin_operator
turn_to satellite0 star6 groundstation3
0
1
0 20 3 7
1
end_operator
begin_operator
turn_to satellite0 star6 phenomenon7
0
1
0 20 4 7
1
end_operator
begin_operator
turn_to satellite0 star6 planet5
0
1
0 20 5 7
1
end_operator
begin_operator
turn_to satellite0 star6 star4
0
1
0 20 6 7
1
end_operator
begin_operator
turn_to satellite1 groundstation0 groundstation1
0
1
0 21 1 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 groundstation2
0
1
0 21 2 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 groundstation3
0
1
0 21 3 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 phenomenon7
0
1
0 21 4 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 planet5
0
1
0 21 5 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 star4
0
1
0 21 6 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 star6
0
1
0 21 7 0
1
end_operator
begin_operator
turn_to satellite1 groundstation1 groundstation0
0
1
0 21 0 1
1
end_operator
begin_operator
turn_to satellite1 groundstation1 groundstation2
0
1
0 21 2 1
1
end_operator
begin_operator
turn_to satellite1 groundstation1 groundstation3
0
1
0 21 3 1
1
end_operator
begin_operator
turn_to satellite1 groundstation1 phenomenon7
0
1
0 21 4 1
1
end_operator
begin_operator
turn_to satellite1 groundstation1 planet5
0
1
0 21 5 1
1
end_operator
begin_operator
turn_to satellite1 groundstation1 star4
0
1
0 21 6 1
1
end_operator
begin_operator
turn_to satellite1 groundstation1 star6
0
1
0 21 7 1
1
end_operator
begin_operator
turn_to satellite1 groundstation2 groundstation0
0
1
0 21 0 2
1
end_operator
begin_operator
turn_to satellite1 groundstation2 groundstation1
0
1
0 21 1 2
1
end_operator
begin_operator
turn_to satellite1 groundstation2 groundstation3
0
1
0 21 3 2
1
end_operator
begin_operator
turn_to satellite1 groundstation2 phenomenon7
0
1
0 21 4 2
1
end_operator
begin_operator
turn_to satellite1 groundstation2 planet5
0
1
0 21 5 2
1
end_operator
begin_operator
turn_to satellite1 groundstation2 star4
0
1
0 21 6 2
1
end_operator
begin_operator
turn_to satellite1 groundstation2 star6
0
1
0 21 7 2
1
end_operator
begin_operator
turn_to satellite1 groundstation3 groundstation0
0
1
0 21 0 3
1
end_operator
begin_operator
turn_to satellite1 groundstation3 groundstation1
0
1
0 21 1 3
1
end_operator
begin_operator
turn_to satellite1 groundstation3 groundstation2
0
1
0 21 2 3
1
end_operator
begin_operator
turn_to satellite1 groundstation3 phenomenon7
0
1
0 21 4 3
1
end_operator
begin_operator
turn_to satellite1 groundstation3 planet5
0
1
0 21 5 3
1
end_operator
begin_operator
turn_to satellite1 groundstation3 star4
0
1
0 21 6 3
1
end_operator
begin_operator
turn_to satellite1 groundstation3 star6
0
1
0 21 7 3
1
end_operator
begin_operator
turn_to satellite1 phenomenon7 groundstation0
0
1
0 21 0 4
1
end_operator
begin_operator
turn_to satellite1 phenomenon7 groundstation1
0
1
0 21 1 4
1
end_operator
begin_operator
turn_to satellite1 phenomenon7 groundstation2
0
1
0 21 2 4
1
end_operator
begin_operator
turn_to satellite1 phenomenon7 groundstation3
0
1
0 21 3 4
1
end_operator
begin_operator
turn_to satellite1 phenomenon7 planet5
0
1
0 21 5 4
1
end_operator
begin_operator
turn_to satellite1 phenomenon7 star4
0
1
0 21 6 4
1
end_operator
begin_operator
turn_to satellite1 phenomenon7 star6
0
1
0 21 7 4
1
end_operator
begin_operator
turn_to satellite1 planet5 groundstation0
0
1
0 21 0 5
1
end_operator
begin_operator
turn_to satellite1 planet5 groundstation1
0
1
0 21 1 5
1
end_operator
begin_operator
turn_to satellite1 planet5 groundstation2
0
1
0 21 2 5
1
end_operator
begin_operator
turn_to satellite1 planet5 groundstation3
0
1
0 21 3 5
1
end_operator
begin_operator
turn_to satellite1 planet5 phenomenon7
0
1
0 21 4 5
1
end_operator
begin_operator
turn_to satellite1 planet5 star4
0
1
0 21 6 5
1
end_operator
begin_operator
turn_to satellite1 planet5 star6
0
1
0 21 7 5
1
end_operator
begin_operator
turn_to satellite1 star4 groundstation0
0
1
0 21 0 6
1
end_operator
begin_operator
turn_to satellite1 star4 groundstation1
0
1
0 21 1 6
1
end_operator
begin_operator
turn_to satellite1 star4 groundstation2
0
1
0 21 2 6
1
end_operator
begin_operator
turn_to satellite1 star4 groundstation3
0
1
0 21 3 6
1
end_operator
begin_operator
turn_to satellite1 star4 phenomenon7
0
1
0 21 4 6
1
end_operator
begin_operator
turn_to satellite1 star4 planet5
0
1
0 21 5 6
1
end_operator
begin_operator
turn_to satellite1 star4 star6
0
1
0 21 7 6
1
end_operator
begin_operator
turn_to satellite1 star6 groundstation0
0
1
0 21 0 7
1
end_operator
begin_operator
turn_to satellite1 star6 groundstation1
0
1
0 21 1 7
1
end_operator
begin_operator
turn_to satellite1 star6 groundstation2
0
1
0 21 2 7
1
end_operator
begin_operator
turn_to satellite1 star6 groundstation3
0
1
0 21 3 7
1
end_operator
begin_operator
turn_to satellite1 star6 phenomenon7
0
1
0 21 4 7
1
end_operator
begin_operator
turn_to satellite1 star6 planet5
0
1
0 21 5 7
1
end_operator
begin_operator
turn_to satellite1 star6 star4
0
1
0 21 6 7
1
end_operator
0
