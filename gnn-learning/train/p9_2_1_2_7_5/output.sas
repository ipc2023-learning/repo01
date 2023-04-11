begin_version
3
end_version
begin_metric
0
end_metric
32
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
Atom have_image(groundstation0, infrared1)
NegatedAtom have_image(groundstation0, infrared1)
end_variable
begin_variable
var3
-1
2
Atom have_image(groundstation0, thermograph0)
NegatedAtom have_image(groundstation0, thermograph0)
end_variable
begin_variable
var4
-1
2
Atom have_image(groundstation1, infrared1)
NegatedAtom have_image(groundstation1, infrared1)
end_variable
begin_variable
var5
-1
2
Atom have_image(groundstation1, thermograph0)
NegatedAtom have_image(groundstation1, thermograph0)
end_variable
begin_variable
var6
-1
2
Atom have_image(groundstation2, infrared1)
NegatedAtom have_image(groundstation2, infrared1)
end_variable
begin_variable
var7
-1
2
Atom have_image(groundstation2, thermograph0)
NegatedAtom have_image(groundstation2, thermograph0)
end_variable
begin_variable
var8
-1
2
Atom have_image(groundstation3, infrared1)
NegatedAtom have_image(groundstation3, infrared1)
end_variable
begin_variable
var9
-1
2
Atom have_image(groundstation3, thermograph0)
NegatedAtom have_image(groundstation3, thermograph0)
end_variable
begin_variable
var10
-1
2
Atom have_image(groundstation5, infrared1)
NegatedAtom have_image(groundstation5, infrared1)
end_variable
begin_variable
var11
-1
2
Atom have_image(groundstation5, thermograph0)
NegatedAtom have_image(groundstation5, thermograph0)
end_variable
begin_variable
var12
-1
2
Atom have_image(groundstation6, infrared1)
NegatedAtom have_image(groundstation6, infrared1)
end_variable
begin_variable
var13
-1
2
Atom have_image(groundstation6, thermograph0)
NegatedAtom have_image(groundstation6, thermograph0)
end_variable
begin_variable
var14
-1
2
Atom have_image(phenomenon10, infrared1)
NegatedAtom have_image(phenomenon10, infrared1)
end_variable
begin_variable
var15
-1
2
Atom have_image(phenomenon10, thermograph0)
NegatedAtom have_image(phenomenon10, thermograph0)
end_variable
begin_variable
var16
-1
2
Atom have_image(phenomenon11, infrared1)
NegatedAtom have_image(phenomenon11, infrared1)
end_variable
begin_variable
var17
-1
2
Atom have_image(phenomenon11, thermograph0)
NegatedAtom have_image(phenomenon11, thermograph0)
end_variable
begin_variable
var18
-1
2
Atom have_image(phenomenon8, infrared1)
NegatedAtom have_image(phenomenon8, infrared1)
end_variable
begin_variable
var19
-1
2
Atom have_image(phenomenon8, thermograph0)
NegatedAtom have_image(phenomenon8, thermograph0)
end_variable
begin_variable
var20
-1
2
Atom have_image(phenomenon9, infrared1)
NegatedAtom have_image(phenomenon9, infrared1)
end_variable
begin_variable
var21
-1
2
Atom have_image(phenomenon9, thermograph0)
NegatedAtom have_image(phenomenon9, thermograph0)
end_variable
begin_variable
var22
-1
2
Atom have_image(planet7, infrared1)
NegatedAtom have_image(planet7, infrared1)
end_variable
begin_variable
var23
-1
2
Atom have_image(planet7, thermograph0)
NegatedAtom have_image(planet7, thermograph0)
end_variable
begin_variable
var24
-1
2
Atom have_image(star4, infrared1)
NegatedAtom have_image(star4, infrared1)
end_variable
begin_variable
var25
-1
2
Atom have_image(star4, thermograph0)
NegatedAtom have_image(star4, thermograph0)
end_variable
begin_variable
var26
-1
12
Atom pointing(satellite0, groundstation0)
Atom pointing(satellite0, groundstation1)
Atom pointing(satellite0, groundstation2)
Atom pointing(satellite0, groundstation3)
Atom pointing(satellite0, groundstation5)
Atom pointing(satellite0, groundstation6)
Atom pointing(satellite0, phenomenon10)
Atom pointing(satellite0, phenomenon11)
Atom pointing(satellite0, phenomenon8)
Atom pointing(satellite0, phenomenon9)
Atom pointing(satellite0, planet7)
Atom pointing(satellite0, star4)
end_variable
begin_variable
var27
-1
12
Atom pointing(satellite1, groundstation0)
Atom pointing(satellite1, groundstation1)
Atom pointing(satellite1, groundstation2)
Atom pointing(satellite1, groundstation3)
Atom pointing(satellite1, groundstation5)
Atom pointing(satellite1, groundstation6)
Atom pointing(satellite1, phenomenon10)
Atom pointing(satellite1, phenomenon11)
Atom pointing(satellite1, phenomenon8)
Atom pointing(satellite1, phenomenon9)
Atom pointing(satellite1, planet7)
Atom pointing(satellite1, star4)
end_variable
begin_variable
var28
-1
2
Atom power_avail(satellite0)
NegatedAtom power_avail(satellite0)
end_variable
begin_variable
var29
-1
2
Atom power_avail(satellite1)
NegatedAtom power_avail(satellite1)
end_variable
begin_variable
var30
-1
2
Atom power_on(instrument0)
NegatedAtom power_on(instrument0)
end_variable
begin_variable
var31
-1
2
Atom power_on(instrument1)
NegatedAtom power_on(instrument1)
end_variable
2
begin_mutex_group
12
26 0
26 1
26 2
26 3
26 4
26 5
26 6
26 7
26 8
26 9
26 10
26 11
end_mutex_group
begin_mutex_group
12
27 0
27 1
27 2
27 3
27 4
27 5
27 6
27 7
27 8
27 9
27 10
27 11
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
1
1
1
1
1
1
9
0
0
0
1
1
end_state
begin_goal
5
15 0
17 0
19 0
21 0
22 0
end_goal
308
begin_operator
calibrate satellite0 instrument0 groundstation1
2
26 1
30 0
1
0 0 -1 0
1
end_operator
begin_operator
calibrate satellite0 instrument0 groundstation2
2
26 2
30 0
1
0 0 -1 0
1
end_operator
begin_operator
calibrate satellite1 instrument1 groundstation5
2
27 4
31 0
1
0 1 -1 0
1
end_operator
begin_operator
calibrate satellite1 instrument1 groundstation6
2
27 5
31 0
1
0 1 -1 0
1
end_operator
begin_operator
switch_off instrument0 satellite0
0
2
0 28 -1 0
0 30 0 1
1
end_operator
begin_operator
switch_off instrument1 satellite1
0
2
0 29 -1 0
0 31 0 1
1
end_operator
begin_operator
switch_on instrument0 satellite0
0
3
0 0 -1 1
0 28 0 1
0 30 -1 0
1
end_operator
begin_operator
switch_on instrument1 satellite1
0
3
0 1 -1 1
0 29 0 1
0 31 -1 0
1
end_operator
begin_operator
take_image satellite0 groundstation0 instrument0 thermograph0
3
0 0
26 0
30 0
1
0 3 -1 0
1
end_operator
begin_operator
take_image satellite0 groundstation1 instrument0 thermograph0
3
0 0
26 1
30 0
1
0 5 -1 0
1
end_operator
begin_operator
take_image satellite0 groundstation2 instrument0 thermograph0
3
0 0
26 2
30 0
1
0 7 -1 0
1
end_operator
begin_operator
take_image satellite0 groundstation3 instrument0 thermograph0
3
0 0
26 3
30 0
1
0 9 -1 0
1
end_operator
begin_operator
take_image satellite0 groundstation5 instrument0 thermograph0
3
0 0
26 4
30 0
1
0 11 -1 0
1
end_operator
begin_operator
take_image satellite0 groundstation6 instrument0 thermograph0
3
0 0
26 5
30 0
1
0 13 -1 0
1
end_operator
begin_operator
take_image satellite0 phenomenon10 instrument0 thermograph0
3
0 0
26 6
30 0
1
0 15 -1 0
1
end_operator
begin_operator
take_image satellite0 phenomenon11 instrument0 thermograph0
3
0 0
26 7
30 0
1
0 17 -1 0
1
end_operator
begin_operator
take_image satellite0 phenomenon8 instrument0 thermograph0
3
0 0
26 8
30 0
1
0 19 -1 0
1
end_operator
begin_operator
take_image satellite0 phenomenon9 instrument0 thermograph0
3
0 0
26 9
30 0
1
0 21 -1 0
1
end_operator
begin_operator
take_image satellite0 planet7 instrument0 thermograph0
3
0 0
26 10
30 0
1
0 23 -1 0
1
end_operator
begin_operator
take_image satellite0 star4 instrument0 thermograph0
3
0 0
26 11
30 0
1
0 25 -1 0
1
end_operator
begin_operator
take_image satellite1 groundstation0 instrument1 infrared1
3
1 0
27 0
31 0
1
0 2 -1 0
1
end_operator
begin_operator
take_image satellite1 groundstation0 instrument1 thermograph0
3
1 0
27 0
31 0
1
0 3 -1 0
1
end_operator
begin_operator
take_image satellite1 groundstation1 instrument1 infrared1
3
1 0
27 1
31 0
1
0 4 -1 0
1
end_operator
begin_operator
take_image satellite1 groundstation1 instrument1 thermograph0
3
1 0
27 1
31 0
1
0 5 -1 0
1
end_operator
begin_operator
take_image satellite1 groundstation2 instrument1 infrared1
3
1 0
27 2
31 0
1
0 6 -1 0
1
end_operator
begin_operator
take_image satellite1 groundstation2 instrument1 thermograph0
3
1 0
27 2
31 0
1
0 7 -1 0
1
end_operator
begin_operator
take_image satellite1 groundstation3 instrument1 infrared1
3
1 0
27 3
31 0
1
0 8 -1 0
1
end_operator
begin_operator
take_image satellite1 groundstation3 instrument1 thermograph0
3
1 0
27 3
31 0
1
0 9 -1 0
1
end_operator
begin_operator
take_image satellite1 groundstation5 instrument1 infrared1
3
1 0
27 4
31 0
1
0 10 -1 0
1
end_operator
begin_operator
take_image satellite1 groundstation5 instrument1 thermograph0
3
1 0
27 4
31 0
1
0 11 -1 0
1
end_operator
begin_operator
take_image satellite1 groundstation6 instrument1 infrared1
3
1 0
27 5
31 0
1
0 12 -1 0
1
end_operator
begin_operator
take_image satellite1 groundstation6 instrument1 thermograph0
3
1 0
27 5
31 0
1
0 13 -1 0
1
end_operator
begin_operator
take_image satellite1 phenomenon10 instrument1 infrared1
3
1 0
27 6
31 0
1
0 14 -1 0
1
end_operator
begin_operator
take_image satellite1 phenomenon10 instrument1 thermograph0
3
1 0
27 6
31 0
1
0 15 -1 0
1
end_operator
begin_operator
take_image satellite1 phenomenon11 instrument1 infrared1
3
1 0
27 7
31 0
1
0 16 -1 0
1
end_operator
begin_operator
take_image satellite1 phenomenon11 instrument1 thermograph0
3
1 0
27 7
31 0
1
0 17 -1 0
1
end_operator
begin_operator
take_image satellite1 phenomenon8 instrument1 infrared1
3
1 0
27 8
31 0
1
0 18 -1 0
1
end_operator
begin_operator
take_image satellite1 phenomenon8 instrument1 thermograph0
3
1 0
27 8
31 0
1
0 19 -1 0
1
end_operator
begin_operator
take_image satellite1 phenomenon9 instrument1 infrared1
3
1 0
27 9
31 0
1
0 20 -1 0
1
end_operator
begin_operator
take_image satellite1 phenomenon9 instrument1 thermograph0
3
1 0
27 9
31 0
1
0 21 -1 0
1
end_operator
begin_operator
take_image satellite1 planet7 instrument1 infrared1
3
1 0
27 10
31 0
1
0 22 -1 0
1
end_operator
begin_operator
take_image satellite1 planet7 instrument1 thermograph0
3
1 0
27 10
31 0
1
0 23 -1 0
1
end_operator
begin_operator
take_image satellite1 star4 instrument1 infrared1
3
1 0
27 11
31 0
1
0 24 -1 0
1
end_operator
begin_operator
take_image satellite1 star4 instrument1 thermograph0
3
1 0
27 11
31 0
1
0 25 -1 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 groundstation1
0
1
0 26 1 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 groundstation2
0
1
0 26 2 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 groundstation3
0
1
0 26 3 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 groundstation5
0
1
0 26 4 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 groundstation6
0
1
0 26 5 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 phenomenon10
0
1
0 26 6 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 phenomenon11
0
1
0 26 7 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 phenomenon8
0
1
0 26 8 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 phenomenon9
0
1
0 26 9 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 planet7
0
1
0 26 10 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 star4
0
1
0 26 11 0
1
end_operator
begin_operator
turn_to satellite0 groundstation1 groundstation0
0
1
0 26 0 1
1
end_operator
begin_operator
turn_to satellite0 groundstation1 groundstation2
0
1
0 26 2 1
1
end_operator
begin_operator
turn_to satellite0 groundstation1 groundstation3
0
1
0 26 3 1
1
end_operator
begin_operator
turn_to satellite0 groundstation1 groundstation5
0
1
0 26 4 1
1
end_operator
begin_operator
turn_to satellite0 groundstation1 groundstation6
0
1
0 26 5 1
1
end_operator
begin_operator
turn_to satellite0 groundstation1 phenomenon10
0
1
0 26 6 1
1
end_operator
begin_operator
turn_to satellite0 groundstation1 phenomenon11
0
1
0 26 7 1
1
end_operator
begin_operator
turn_to satellite0 groundstation1 phenomenon8
0
1
0 26 8 1
1
end_operator
begin_operator
turn_to satellite0 groundstation1 phenomenon9
0
1
0 26 9 1
1
end_operator
begin_operator
turn_to satellite0 groundstation1 planet7
0
1
0 26 10 1
1
end_operator
begin_operator
turn_to satellite0 groundstation1 star4
0
1
0 26 11 1
1
end_operator
begin_operator
turn_to satellite0 groundstation2 groundstation0
0
1
0 26 0 2
1
end_operator
begin_operator
turn_to satellite0 groundstation2 groundstation1
0
1
0 26 1 2
1
end_operator
begin_operator
turn_to satellite0 groundstation2 groundstation3
0
1
0 26 3 2
1
end_operator
begin_operator
turn_to satellite0 groundstation2 groundstation5
0
1
0 26 4 2
1
end_operator
begin_operator
turn_to satellite0 groundstation2 groundstation6
0
1
0 26 5 2
1
end_operator
begin_operator
turn_to satellite0 groundstation2 phenomenon10
0
1
0 26 6 2
1
end_operator
begin_operator
turn_to satellite0 groundstation2 phenomenon11
0
1
0 26 7 2
1
end_operator
begin_operator
turn_to satellite0 groundstation2 phenomenon8
0
1
0 26 8 2
1
end_operator
begin_operator
turn_to satellite0 groundstation2 phenomenon9
0
1
0 26 9 2
1
end_operator
begin_operator
turn_to satellite0 groundstation2 planet7
0
1
0 26 10 2
1
end_operator
begin_operator
turn_to satellite0 groundstation2 star4
0
1
0 26 11 2
1
end_operator
begin_operator
turn_to satellite0 groundstation3 groundstation0
0
1
0 26 0 3
1
end_operator
begin_operator
turn_to satellite0 groundstation3 groundstation1
0
1
0 26 1 3
1
end_operator
begin_operator
turn_to satellite0 groundstation3 groundstation2
0
1
0 26 2 3
1
end_operator
begin_operator
turn_to satellite0 groundstation3 groundstation5
0
1
0 26 4 3
1
end_operator
begin_operator
turn_to satellite0 groundstation3 groundstation6
0
1
0 26 5 3
1
end_operator
begin_operator
turn_to satellite0 groundstation3 phenomenon10
0
1
0 26 6 3
1
end_operator
begin_operator
turn_to satellite0 groundstation3 phenomenon11
0
1
0 26 7 3
1
end_operator
begin_operator
turn_to satellite0 groundstation3 phenomenon8
0
1
0 26 8 3
1
end_operator
begin_operator
turn_to satellite0 groundstation3 phenomenon9
0
1
0 26 9 3
1
end_operator
begin_operator
turn_to satellite0 groundstation3 planet7
0
1
0 26 10 3
1
end_operator
begin_operator
turn_to satellite0 groundstation3 star4
0
1
0 26 11 3
1
end_operator
begin_operator
turn_to satellite0 groundstation5 groundstation0
0
1
0 26 0 4
1
end_operator
begin_operator
turn_to satellite0 groundstation5 groundstation1
0
1
0 26 1 4
1
end_operator
begin_operator
turn_to satellite0 groundstation5 groundstation2
0
1
0 26 2 4
1
end_operator
begin_operator
turn_to satellite0 groundstation5 groundstation3
0
1
0 26 3 4
1
end_operator
begin_operator
turn_to satellite0 groundstation5 groundstation6
0
1
0 26 5 4
1
end_operator
begin_operator
turn_to satellite0 groundstation5 phenomenon10
0
1
0 26 6 4
1
end_operator
begin_operator
turn_to satellite0 groundstation5 phenomenon11
0
1
0 26 7 4
1
end_operator
begin_operator
turn_to satellite0 groundstation5 phenomenon8
0
1
0 26 8 4
1
end_operator
begin_operator
turn_to satellite0 groundstation5 phenomenon9
0
1
0 26 9 4
1
end_operator
begin_operator
turn_to satellite0 groundstation5 planet7
0
1
0 26 10 4
1
end_operator
begin_operator
turn_to satellite0 groundstation5 star4
0
1
0 26 11 4
1
end_operator
begin_operator
turn_to satellite0 groundstation6 groundstation0
0
1
0 26 0 5
1
end_operator
begin_operator
turn_to satellite0 groundstation6 groundstation1
0
1
0 26 1 5
1
end_operator
begin_operator
turn_to satellite0 groundstation6 groundstation2
0
1
0 26 2 5
1
end_operator
begin_operator
turn_to satellite0 groundstation6 groundstation3
0
1
0 26 3 5
1
end_operator
begin_operator
turn_to satellite0 groundstation6 groundstation5
0
1
0 26 4 5
1
end_operator
begin_operator
turn_to satellite0 groundstation6 phenomenon10
0
1
0 26 6 5
1
end_operator
begin_operator
turn_to satellite0 groundstation6 phenomenon11
0
1
0 26 7 5
1
end_operator
begin_operator
turn_to satellite0 groundstation6 phenomenon8
0
1
0 26 8 5
1
end_operator
begin_operator
turn_to satellite0 groundstation6 phenomenon9
0
1
0 26 9 5
1
end_operator
begin_operator
turn_to satellite0 groundstation6 planet7
0
1
0 26 10 5
1
end_operator
begin_operator
turn_to satellite0 groundstation6 star4
0
1
0 26 11 5
1
end_operator
begin_operator
turn_to satellite0 phenomenon10 groundstation0
0
1
0 26 0 6
1
end_operator
begin_operator
turn_to satellite0 phenomenon10 groundstation1
0
1
0 26 1 6
1
end_operator
begin_operator
turn_to satellite0 phenomenon10 groundstation2
0
1
0 26 2 6
1
end_operator
begin_operator
turn_to satellite0 phenomenon10 groundstation3
0
1
0 26 3 6
1
end_operator
begin_operator
turn_to satellite0 phenomenon10 groundstation5
0
1
0 26 4 6
1
end_operator
begin_operator
turn_to satellite0 phenomenon10 groundstation6
0
1
0 26 5 6
1
end_operator
begin_operator
turn_to satellite0 phenomenon10 phenomenon11
0
1
0 26 7 6
1
end_operator
begin_operator
turn_to satellite0 phenomenon10 phenomenon8
0
1
0 26 8 6
1
end_operator
begin_operator
turn_to satellite0 phenomenon10 phenomenon9
0
1
0 26 9 6
1
end_operator
begin_operator
turn_to satellite0 phenomenon10 planet7
0
1
0 26 10 6
1
end_operator
begin_operator
turn_to satellite0 phenomenon10 star4
0
1
0 26 11 6
1
end_operator
begin_operator
turn_to satellite0 phenomenon11 groundstation0
0
1
0 26 0 7
1
end_operator
begin_operator
turn_to satellite0 phenomenon11 groundstation1
0
1
0 26 1 7
1
end_operator
begin_operator
turn_to satellite0 phenomenon11 groundstation2
0
1
0 26 2 7
1
end_operator
begin_operator
turn_to satellite0 phenomenon11 groundstation3
0
1
0 26 3 7
1
end_operator
begin_operator
turn_to satellite0 phenomenon11 groundstation5
0
1
0 26 4 7
1
end_operator
begin_operator
turn_to satellite0 phenomenon11 groundstation6
0
1
0 26 5 7
1
end_operator
begin_operator
turn_to satellite0 phenomenon11 phenomenon10
0
1
0 26 6 7
1
end_operator
begin_operator
turn_to satellite0 phenomenon11 phenomenon8
0
1
0 26 8 7
1
end_operator
begin_operator
turn_to satellite0 phenomenon11 phenomenon9
0
1
0 26 9 7
1
end_operator
begin_operator
turn_to satellite0 phenomenon11 planet7
0
1
0 26 10 7
1
end_operator
begin_operator
turn_to satellite0 phenomenon11 star4
0
1
0 26 11 7
1
end_operator
begin_operator
turn_to satellite0 phenomenon8 groundstation0
0
1
0 26 0 8
1
end_operator
begin_operator
turn_to satellite0 phenomenon8 groundstation1
0
1
0 26 1 8
1
end_operator
begin_operator
turn_to satellite0 phenomenon8 groundstation2
0
1
0 26 2 8
1
end_operator
begin_operator
turn_to satellite0 phenomenon8 groundstation3
0
1
0 26 3 8
1
end_operator
begin_operator
turn_to satellite0 phenomenon8 groundstation5
0
1
0 26 4 8
1
end_operator
begin_operator
turn_to satellite0 phenomenon8 groundstation6
0
1
0 26 5 8
1
end_operator
begin_operator
turn_to satellite0 phenomenon8 phenomenon10
0
1
0 26 6 8
1
end_operator
begin_operator
turn_to satellite0 phenomenon8 phenomenon11
0
1
0 26 7 8
1
end_operator
begin_operator
turn_to satellite0 phenomenon8 phenomenon9
0
1
0 26 9 8
1
end_operator
begin_operator
turn_to satellite0 phenomenon8 planet7
0
1
0 26 10 8
1
end_operator
begin_operator
turn_to satellite0 phenomenon8 star4
0
1
0 26 11 8
1
end_operator
begin_operator
turn_to satellite0 phenomenon9 groundstation0
0
1
0 26 0 9
1
end_operator
begin_operator
turn_to satellite0 phenomenon9 groundstation1
0
1
0 26 1 9
1
end_operator
begin_operator
turn_to satellite0 phenomenon9 groundstation2
0
1
0 26 2 9
1
end_operator
begin_operator
turn_to satellite0 phenomenon9 groundstation3
0
1
0 26 3 9
1
end_operator
begin_operator
turn_to satellite0 phenomenon9 groundstation5
0
1
0 26 4 9
1
end_operator
begin_operator
turn_to satellite0 phenomenon9 groundstation6
0
1
0 26 5 9
1
end_operator
begin_operator
turn_to satellite0 phenomenon9 phenomenon10
0
1
0 26 6 9
1
end_operator
begin_operator
turn_to satellite0 phenomenon9 phenomenon11
0
1
0 26 7 9
1
end_operator
begin_operator
turn_to satellite0 phenomenon9 phenomenon8
0
1
0 26 8 9
1
end_operator
begin_operator
turn_to satellite0 phenomenon9 planet7
0
1
0 26 10 9
1
end_operator
begin_operator
turn_to satellite0 phenomenon9 star4
0
1
0 26 11 9
1
end_operator
begin_operator
turn_to satellite0 planet7 groundstation0
0
1
0 26 0 10
1
end_operator
begin_operator
turn_to satellite0 planet7 groundstation1
0
1
0 26 1 10
1
end_operator
begin_operator
turn_to satellite0 planet7 groundstation2
0
1
0 26 2 10
1
end_operator
begin_operator
turn_to satellite0 planet7 groundstation3
0
1
0 26 3 10
1
end_operator
begin_operator
turn_to satellite0 planet7 groundstation5
0
1
0 26 4 10
1
end_operator
begin_operator
turn_to satellite0 planet7 groundstation6
0
1
0 26 5 10
1
end_operator
begin_operator
turn_to satellite0 planet7 phenomenon10
0
1
0 26 6 10
1
end_operator
begin_operator
turn_to satellite0 planet7 phenomenon11
0
1
0 26 7 10
1
end_operator
begin_operator
turn_to satellite0 planet7 phenomenon8
0
1
0 26 8 10
1
end_operator
begin_operator
turn_to satellite0 planet7 phenomenon9
0
1
0 26 9 10
1
end_operator
begin_operator
turn_to satellite0 planet7 star4
0
1
0 26 11 10
1
end_operator
begin_operator
turn_to satellite0 star4 groundstation0
0
1
0 26 0 11
1
end_operator
begin_operator
turn_to satellite0 star4 groundstation1
0
1
0 26 1 11
1
end_operator
begin_operator
turn_to satellite0 star4 groundstation2
0
1
0 26 2 11
1
end_operator
begin_operator
turn_to satellite0 star4 groundstation3
0
1
0 26 3 11
1
end_operator
begin_operator
turn_to satellite0 star4 groundstation5
0
1
0 26 4 11
1
end_operator
begin_operator
turn_to satellite0 star4 groundstation6
0
1
0 26 5 11
1
end_operator
begin_operator
turn_to satellite0 star4 phenomenon10
0
1
0 26 6 11
1
end_operator
begin_operator
turn_to satellite0 star4 phenomenon11
0
1
0 26 7 11
1
end_operator
begin_operator
turn_to satellite0 star4 phenomenon8
0
1
0 26 8 11
1
end_operator
begin_operator
turn_to satellite0 star4 phenomenon9
0
1
0 26 9 11
1
end_operator
begin_operator
turn_to satellite0 star4 planet7
0
1
0 26 10 11
1
end_operator
begin_operator
turn_to satellite1 groundstation0 groundstation1
0
1
0 27 1 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 groundstation2
0
1
0 27 2 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 groundstation3
0
1
0 27 3 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 groundstation5
0
1
0 27 4 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 groundstation6
0
1
0 27 5 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 phenomenon10
0
1
0 27 6 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 phenomenon11
0
1
0 27 7 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 phenomenon8
0
1
0 27 8 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 phenomenon9
0
1
0 27 9 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 planet7
0
1
0 27 10 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 star4
0
1
0 27 11 0
1
end_operator
begin_operator
turn_to satellite1 groundstation1 groundstation0
0
1
0 27 0 1
1
end_operator
begin_operator
turn_to satellite1 groundstation1 groundstation2
0
1
0 27 2 1
1
end_operator
begin_operator
turn_to satellite1 groundstation1 groundstation3
0
1
0 27 3 1
1
end_operator
begin_operator
turn_to satellite1 groundstation1 groundstation5
0
1
0 27 4 1
1
end_operator
begin_operator
turn_to satellite1 groundstation1 groundstation6
0
1
0 27 5 1
1
end_operator
begin_operator
turn_to satellite1 groundstation1 phenomenon10
0
1
0 27 6 1
1
end_operator
begin_operator
turn_to satellite1 groundstation1 phenomenon11
0
1
0 27 7 1
1
end_operator
begin_operator
turn_to satellite1 groundstation1 phenomenon8
0
1
0 27 8 1
1
end_operator
begin_operator
turn_to satellite1 groundstation1 phenomenon9
0
1
0 27 9 1
1
end_operator
begin_operator
turn_to satellite1 groundstation1 planet7
0
1
0 27 10 1
1
end_operator
begin_operator
turn_to satellite1 groundstation1 star4
0
1
0 27 11 1
1
end_operator
begin_operator
turn_to satellite1 groundstation2 groundstation0
0
1
0 27 0 2
1
end_operator
begin_operator
turn_to satellite1 groundstation2 groundstation1
0
1
0 27 1 2
1
end_operator
begin_operator
turn_to satellite1 groundstation2 groundstation3
0
1
0 27 3 2
1
end_operator
begin_operator
turn_to satellite1 groundstation2 groundstation5
0
1
0 27 4 2
1
end_operator
begin_operator
turn_to satellite1 groundstation2 groundstation6
0
1
0 27 5 2
1
end_operator
begin_operator
turn_to satellite1 groundstation2 phenomenon10
0
1
0 27 6 2
1
end_operator
begin_operator
turn_to satellite1 groundstation2 phenomenon11
0
1
0 27 7 2
1
end_operator
begin_operator
turn_to satellite1 groundstation2 phenomenon8
0
1
0 27 8 2
1
end_operator
begin_operator
turn_to satellite1 groundstation2 phenomenon9
0
1
0 27 9 2
1
end_operator
begin_operator
turn_to satellite1 groundstation2 planet7
0
1
0 27 10 2
1
end_operator
begin_operator
turn_to satellite1 groundstation2 star4
0
1
0 27 11 2
1
end_operator
begin_operator
turn_to satellite1 groundstation3 groundstation0
0
1
0 27 0 3
1
end_operator
begin_operator
turn_to satellite1 groundstation3 groundstation1
0
1
0 27 1 3
1
end_operator
begin_operator
turn_to satellite1 groundstation3 groundstation2
0
1
0 27 2 3
1
end_operator
begin_operator
turn_to satellite1 groundstation3 groundstation5
0
1
0 27 4 3
1
end_operator
begin_operator
turn_to satellite1 groundstation3 groundstation6
0
1
0 27 5 3
1
end_operator
begin_operator
turn_to satellite1 groundstation3 phenomenon10
0
1
0 27 6 3
1
end_operator
begin_operator
turn_to satellite1 groundstation3 phenomenon11
0
1
0 27 7 3
1
end_operator
begin_operator
turn_to satellite1 groundstation3 phenomenon8
0
1
0 27 8 3
1
end_operator
begin_operator
turn_to satellite1 groundstation3 phenomenon9
0
1
0 27 9 3
1
end_operator
begin_operator
turn_to satellite1 groundstation3 planet7
0
1
0 27 10 3
1
end_operator
begin_operator
turn_to satellite1 groundstation3 star4
0
1
0 27 11 3
1
end_operator
begin_operator
turn_to satellite1 groundstation5 groundstation0
0
1
0 27 0 4
1
end_operator
begin_operator
turn_to satellite1 groundstation5 groundstation1
0
1
0 27 1 4
1
end_operator
begin_operator
turn_to satellite1 groundstation5 groundstation2
0
1
0 27 2 4
1
end_operator
begin_operator
turn_to satellite1 groundstation5 groundstation3
0
1
0 27 3 4
1
end_operator
begin_operator
turn_to satellite1 groundstation5 groundstation6
0
1
0 27 5 4
1
end_operator
begin_operator
turn_to satellite1 groundstation5 phenomenon10
0
1
0 27 6 4
1
end_operator
begin_operator
turn_to satellite1 groundstation5 phenomenon11
0
1
0 27 7 4
1
end_operator
begin_operator
turn_to satellite1 groundstation5 phenomenon8
0
1
0 27 8 4
1
end_operator
begin_operator
turn_to satellite1 groundstation5 phenomenon9
0
1
0 27 9 4
1
end_operator
begin_operator
turn_to satellite1 groundstation5 planet7
0
1
0 27 10 4
1
end_operator
begin_operator
turn_to satellite1 groundstation5 star4
0
1
0 27 11 4
1
end_operator
begin_operator
turn_to satellite1 groundstation6 groundstation0
0
1
0 27 0 5
1
end_operator
begin_operator
turn_to satellite1 groundstation6 groundstation1
0
1
0 27 1 5
1
end_operator
begin_operator
turn_to satellite1 groundstation6 groundstation2
0
1
0 27 2 5
1
end_operator
begin_operator
turn_to satellite1 groundstation6 groundstation3
0
1
0 27 3 5
1
end_operator
begin_operator
turn_to satellite1 groundstation6 groundstation5
0
1
0 27 4 5
1
end_operator
begin_operator
turn_to satellite1 groundstation6 phenomenon10
0
1
0 27 6 5
1
end_operator
begin_operator
turn_to satellite1 groundstation6 phenomenon11
0
1
0 27 7 5
1
end_operator
begin_operator
turn_to satellite1 groundstation6 phenomenon8
0
1
0 27 8 5
1
end_operator
begin_operator
turn_to satellite1 groundstation6 phenomenon9
0
1
0 27 9 5
1
end_operator
begin_operator
turn_to satellite1 groundstation6 planet7
0
1
0 27 10 5
1
end_operator
begin_operator
turn_to satellite1 groundstation6 star4
0
1
0 27 11 5
1
end_operator
begin_operator
turn_to satellite1 phenomenon10 groundstation0
0
1
0 27 0 6
1
end_operator
begin_operator
turn_to satellite1 phenomenon10 groundstation1
0
1
0 27 1 6
1
end_operator
begin_operator
turn_to satellite1 phenomenon10 groundstation2
0
1
0 27 2 6
1
end_operator
begin_operator
turn_to satellite1 phenomenon10 groundstation3
0
1
0 27 3 6
1
end_operator
begin_operator
turn_to satellite1 phenomenon10 groundstation5
0
1
0 27 4 6
1
end_operator
begin_operator
turn_to satellite1 phenomenon10 groundstation6
0
1
0 27 5 6
1
end_operator
begin_operator
turn_to satellite1 phenomenon10 phenomenon11
0
1
0 27 7 6
1
end_operator
begin_operator
turn_to satellite1 phenomenon10 phenomenon8
0
1
0 27 8 6
1
end_operator
begin_operator
turn_to satellite1 phenomenon10 phenomenon9
0
1
0 27 9 6
1
end_operator
begin_operator
turn_to satellite1 phenomenon10 planet7
0
1
0 27 10 6
1
end_operator
begin_operator
turn_to satellite1 phenomenon10 star4
0
1
0 27 11 6
1
end_operator
begin_operator
turn_to satellite1 phenomenon11 groundstation0
0
1
0 27 0 7
1
end_operator
begin_operator
turn_to satellite1 phenomenon11 groundstation1
0
1
0 27 1 7
1
end_operator
begin_operator
turn_to satellite1 phenomenon11 groundstation2
0
1
0 27 2 7
1
end_operator
begin_operator
turn_to satellite1 phenomenon11 groundstation3
0
1
0 27 3 7
1
end_operator
begin_operator
turn_to satellite1 phenomenon11 groundstation5
0
1
0 27 4 7
1
end_operator
begin_operator
turn_to satellite1 phenomenon11 groundstation6
0
1
0 27 5 7
1
end_operator
begin_operator
turn_to satellite1 phenomenon11 phenomenon10
0
1
0 27 6 7
1
end_operator
begin_operator
turn_to satellite1 phenomenon11 phenomenon8
0
1
0 27 8 7
1
end_operator
begin_operator
turn_to satellite1 phenomenon11 phenomenon9
0
1
0 27 9 7
1
end_operator
begin_operator
turn_to satellite1 phenomenon11 planet7
0
1
0 27 10 7
1
end_operator
begin_operator
turn_to satellite1 phenomenon11 star4
0
1
0 27 11 7
1
end_operator
begin_operator
turn_to satellite1 phenomenon8 groundstation0
0
1
0 27 0 8
1
end_operator
begin_operator
turn_to satellite1 phenomenon8 groundstation1
0
1
0 27 1 8
1
end_operator
begin_operator
turn_to satellite1 phenomenon8 groundstation2
0
1
0 27 2 8
1
end_operator
begin_operator
turn_to satellite1 phenomenon8 groundstation3
0
1
0 27 3 8
1
end_operator
begin_operator
turn_to satellite1 phenomenon8 groundstation5
0
1
0 27 4 8
1
end_operator
begin_operator
turn_to satellite1 phenomenon8 groundstation6
0
1
0 27 5 8
1
end_operator
begin_operator
turn_to satellite1 phenomenon8 phenomenon10
0
1
0 27 6 8
1
end_operator
begin_operator
turn_to satellite1 phenomenon8 phenomenon11
0
1
0 27 7 8
1
end_operator
begin_operator
turn_to satellite1 phenomenon8 phenomenon9
0
1
0 27 9 8
1
end_operator
begin_operator
turn_to satellite1 phenomenon8 planet7
0
1
0 27 10 8
1
end_operator
begin_operator
turn_to satellite1 phenomenon8 star4
0
1
0 27 11 8
1
end_operator
begin_operator
turn_to satellite1 phenomenon9 groundstation0
0
1
0 27 0 9
1
end_operator
begin_operator
turn_to satellite1 phenomenon9 groundstation1
0
1
0 27 1 9
1
end_operator
begin_operator
turn_to satellite1 phenomenon9 groundstation2
0
1
0 27 2 9
1
end_operator
begin_operator
turn_to satellite1 phenomenon9 groundstation3
0
1
0 27 3 9
1
end_operator
begin_operator
turn_to satellite1 phenomenon9 groundstation5
0
1
0 27 4 9
1
end_operator
begin_operator
turn_to satellite1 phenomenon9 groundstation6
0
1
0 27 5 9
1
end_operator
begin_operator
turn_to satellite1 phenomenon9 phenomenon10
0
1
0 27 6 9
1
end_operator
begin_operator
turn_to satellite1 phenomenon9 phenomenon11
0
1
0 27 7 9
1
end_operator
begin_operator
turn_to satellite1 phenomenon9 phenomenon8
0
1
0 27 8 9
1
end_operator
begin_operator
turn_to satellite1 phenomenon9 planet7
0
1
0 27 10 9
1
end_operator
begin_operator
turn_to satellite1 phenomenon9 star4
0
1
0 27 11 9
1
end_operator
begin_operator
turn_to satellite1 planet7 groundstation0
0
1
0 27 0 10
1
end_operator
begin_operator
turn_to satellite1 planet7 groundstation1
0
1
0 27 1 10
1
end_operator
begin_operator
turn_to satellite1 planet7 groundstation2
0
1
0 27 2 10
1
end_operator
begin_operator
turn_to satellite1 planet7 groundstation3
0
1
0 27 3 10
1
end_operator
begin_operator
turn_to satellite1 planet7 groundstation5
0
1
0 27 4 10
1
end_operator
begin_operator
turn_to satellite1 planet7 groundstation6
0
1
0 27 5 10
1
end_operator
begin_operator
turn_to satellite1 planet7 phenomenon10
0
1
0 27 6 10
1
end_operator
begin_operator
turn_to satellite1 planet7 phenomenon11
0
1
0 27 7 10
1
end_operator
begin_operator
turn_to satellite1 planet7 phenomenon8
0
1
0 27 8 10
1
end_operator
begin_operator
turn_to satellite1 planet7 phenomenon9
0
1
0 27 9 10
1
end_operator
begin_operator
turn_to satellite1 planet7 star4
0
1
0 27 11 10
1
end_operator
begin_operator
turn_to satellite1 star4 groundstation0
0
1
0 27 0 11
1
end_operator
begin_operator
turn_to satellite1 star4 groundstation1
0
1
0 27 1 11
1
end_operator
begin_operator
turn_to satellite1 star4 groundstation2
0
1
0 27 2 11
1
end_operator
begin_operator
turn_to satellite1 star4 groundstation3
0
1
0 27 3 11
1
end_operator
begin_operator
turn_to satellite1 star4 groundstation5
0
1
0 27 4 11
1
end_operator
begin_operator
turn_to satellite1 star4 groundstation6
0
1
0 27 5 11
1
end_operator
begin_operator
turn_to satellite1 star4 phenomenon10
0
1
0 27 6 11
1
end_operator
begin_operator
turn_to satellite1 star4 phenomenon11
0
1
0 27 7 11
1
end_operator
begin_operator
turn_to satellite1 star4 phenomenon8
0
1
0 27 8 11
1
end_operator
begin_operator
turn_to satellite1 star4 phenomenon9
0
1
0 27 9 11
1
end_operator
begin_operator
turn_to satellite1 star4 planet7
0
1
0 27 10 11
1
end_operator
0
